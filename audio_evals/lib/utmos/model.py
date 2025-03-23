import torch
import torch.nn as nn
import fairseq
import os
import hydra


def load_ssl_model(cp_path):
    ssl_model_type = cp_path.split("/")[-1]
    wavlm = "WavLM" in ssl_model_type
    if wavlm:
        checkpoint = torch.load(cp_path)
        cfg = WavLMConfig(checkpoint["cfg"])
        ssl_model = WavLM(cfg)
        ssl_model.load_state_dict(checkpoint["model"])
        if "Large" in ssl_model_type:
            SSL_OUT_DIM = 1024
        else:
            SSL_OUT_DIM = 768
    else:
        if ssl_model_type == "wav2vec_small.pt":
            SSL_OUT_DIM = 768
        elif ssl_model_type in ["w2v_large_lv_fsh_swbd_cv.pt", "xlsr_53_56k.pt"]:
            SSL_OUT_DIM = 1024
        else:
            print("*** ERROR *** SSL model type " + ssl_model_type + " not supported.")
            exit()
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
            [cp_path]
        )
        ssl_model = model[0]
        ssl_model.remove_pretraining_modules()
    return SSL_model(ssl_model, SSL_OUT_DIM, wavlm)


class SSL_model(nn.Module):
    def __init__(self, ssl_model, ssl_out_dim, wavlm) -> None:
        super(SSL_model, self).__init__()
        self.ssl_model, self.ssl_out_dim = ssl_model, ssl_out_dim
        self.WavLM = wavlm

    def forward(self, batch):
        wav = batch["wav"]
        wav = wav.squeeze(1)  # [batches, audio_len]
        if self.WavLM:
            x = self.ssl_model.extract_features(wav)[0]
        else:
            res = self.ssl_model(wav, mask=False, features_only=True)
            x = res["x"]
        return {"ssl-feature": x}

    def get_output_dim(self):
        return self.ssl_out_dim


class PhonemeEncoder(nn.Module):
    """
    PhonemeEncoder consists of an embedding layer, an LSTM layer, and a linear layer.
    Args:
        vocab_size: the size of the vocabulary
        hidden_dim: the size of the hidden state of the LSTM
        emb_dim: the size of the embedding layer
        out_dim: the size of the output of the linear layer
        n_lstm_layers: the number of LSTM layers
    """

    def __init__(
        self,
        vocab_size,
        hidden_dim,
        emb_dim,
        out_dim,
        n_lstm_layers,
        with_reference=True,
    ) -> None:
        super().__init__()
        self.with_reference = with_reference
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.encoder = nn.LSTM(
            emb_dim,
            hidden_dim,
            num_layers=n_lstm_layers,
            dropout=0.1,
            bidirectional=True,
        )
        self.linear = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim * self.with_reference, out_dim), nn.ReLU()
        )
        self.out_dim = out_dim

    def forward(self, batch):
        seq = batch["phonemes"]
        lens = batch["phoneme_lens"]
        reference_seq = batch["reference"]
        reference_lens = batch["reference_lens"]
        emb = self.embedding(seq)
        emb = torch.nn.utils.rnn.pack_padded_sequence(
            emb, lens, batch_first=True, enforce_sorted=False
        )
        _, (ht, _) = self.encoder(emb)
        feature = ht[-1] + ht[0]
        if self.with_reference:
            if reference_seq == None or reference_lens == None:
                raise ValueError(
                    "reference_batch and reference_lens should not be None when with_reference is True"
                )
            reference_emb = self.embedding(reference_seq)
            reference_emb = torch.nn.utils.rnn.pack_padded_sequence(
                reference_emb, reference_lens, batch_first=True, enforce_sorted=False
            )
            _, (ht_ref, _) = self.encoder(emb)
            reference_feature = ht_ref[-1] + ht_ref[0]
            feature = self.linear(torch.cat([feature, reference_feature], 1))
        else:
            feature = self.linear(feature)
        return {"phoneme-feature": feature}

    def get_output_dim(self):
        return self.out_dim


class DomainEmbedding(nn.Module):
    def __init__(self, n_domains, domain_dim) -> None:
        super().__init__()
        self.embedding = nn.Embedding(n_domains, domain_dim)
        self.output_dim = domain_dim

    def forward(self, batch):
        return {"domain-feature": self.embedding(batch["domains"])}

    def get_output_dim(self):
        return self.output_dim


class LDConditioner(nn.Module):
    """
    Conditions ssl output by listener embedding
    """

    def __init__(self, input_dim, judge_dim, num_judges=None):
        super().__init__()
        self.input_dim = input_dim
        self.judge_dim = judge_dim
        self.num_judges = num_judges
        assert num_judges != None
        self.judge_embedding = nn.Embedding(num_judges, self.judge_dim)
        # concat [self.output_layer, phoneme features]

        self.decoder_rnn = nn.LSTM(
            input_size=self.input_dim + self.judge_dim,
            hidden_size=512,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )  # linear?
        self.out_dim = self.decoder_rnn.hidden_size * 2

    def get_output_dim(self):
        return self.out_dim

    def forward(self, x, batch):
        judge_ids = batch["judge_id"]
        if "phoneme-feature" in x.keys():
            concatenated_feature = torch.cat(
                (
                    x["ssl-feature"],
                    x["phoneme-feature"]
                    .unsqueeze(1)
                    .expand(-1, x["ssl-feature"].size(1), -1),
                ),
                dim=2,
            )
        else:
            concatenated_feature = x["ssl-feature"]
        if "domain-feature" in x.keys():
            concatenated_feature = torch.cat(
                (
                    concatenated_feature,
                    x["domain-feature"]
                    .unsqueeze(1)
                    .expand(-1, concatenated_feature.size(1), -1),
                ),
                dim=2,
            )
        if judge_ids != None:
            concatenated_feature = torch.cat(
                (
                    concatenated_feature,
                    self.judge_embedding(judge_ids)
                    .unsqueeze(1)
                    .expand(-1, concatenated_feature.size(1), -1),
                ),
                dim=2,
            )
            decoder_output, (h, c) = self.decoder_rnn(concatenated_feature)
        return decoder_output


class Projection(nn.Module):
    def __init__(self, input_dim, hidden_dim, activation, range_clipping=False):
        super(Projection, self).__init__()
        self.range_clipping = range_clipping
        output_dim = 1
        if range_clipping:
            self.proj = nn.Tanh()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            activation,
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim),
        )
        self.output_dim = output_dim

    def forward(self, x, batch):
        output = self.net(x)

        # range clipping
        if self.range_clipping:
            return self.proj(output) * 2.0 + 3
        else:
            return output

    def get_output_dim(self):
        return self.output_dim
