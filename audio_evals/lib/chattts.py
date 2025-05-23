# ChatTTS Audio tokenizer

import math
from typing import List, Optional, Literal, Union
import numpy as np
import torch
import torch.nn as nn
import torchaudio
from vector_quantize_pytorch import GroupedResidualFSQ

from vocos import Vocos
from vocos.pretrained import instantiate_class

from dataclasses import dataclass, asdict


class ConvNeXtBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        kernel: int,
        dilation: int,
        layer_scale_init_value: float = 1e-6,
    ):
        # ConvNeXt Block copied from Vocos.
        super().__init__()
        self.dwconv = nn.Conv1d(
            dim,
            dim,
            kernel_size=kernel,
            padding=dilation * (kernel // 2),
            dilation=dilation,
            groups=dim,
        )  # depthwise conv

        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(
            dim, intermediate_dim
        )  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(intermediate_dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )

    def forward(self, x: torch.Tensor, cond=None) -> torch.Tensor:
        residual = x

        y = self.dwconv(x)
        y.transpose_(1, 2)  # (B, C, T) -> (B, T, C)
        x = self.norm(y)
        del y
        y = self.pwconv1(x)
        del x
        x = self.act(y)
        del y
        y = self.pwconv2(x)
        del x
        if self.gamma is not None:
            y *= self.gamma
        y.transpose_(1, 2)  # (B, T, C) -> (B, C, T)

        x = y + residual
        del y

        return x


class GFSQ(nn.Module):

    def __init__(
        self, dim: int, levels: List[int], G: int, R: int, eps=1e-5, transpose=True
    ):
        super(GFSQ, self).__init__()
        self.quantizer = GroupedResidualFSQ(
            dim=dim,
            levels=list(levels),
            num_quantizers=R,
            groups=G,
        )
        self.n_ind = math.prod(levels)
        self.eps = eps
        self.transpose = transpose
        self.G = G
        self.R = R

    def _embed(self, x: torch.Tensor):
        if self.transpose:
            x = x.transpose(1, 2)
        """
        x = rearrange(
            x, "b t (g r) -> g b t r", g = self.G, r = self.R,
        )
        """
        x = x.view(x.size(0), x.size(1), self.G, self.R).permute(2, 0, 1, 3)
        feat = self.quantizer.get_output_from_indices(x)
        return feat.transpose_(1, 2) if self.transpose else feat

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return super().__call__(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.transpose:
            x.transpose_(1, 2)
        # feat, ind = self.quantizer(x)
        _, ind = self.quantizer(x)
        """
        ind = rearrange(
            ind, "g b t r ->b t (g r)",
        )
        """
        ind = ind.permute(1, 2, 0, 3).contiguous()
        ind = ind.view(ind.size(0), ind.size(1), -1)
        """
        embed_onehot_tmp = F.one_hot(ind.long(), self.n_ind)
        embed_onehot = embed_onehot_tmp.to(x.dtype)
        del embed_onehot_tmp
        e_mean = torch.mean(embed_onehot, dim=[0, 1])
        # e_mean = e_mean / (e_mean.sum(dim=1) + self.eps).unsqueeze(1)
        torch.div(e_mean, (e_mean.sum(dim=1) + self.eps).unsqueeze(1), out=e_mean)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + self.eps), dim=1))

        return
            torch.zeros(perplexity.shape, dtype=x.dtype, device=x.device),
            feat.transpose_(1, 2) if self.transpose else feat,
            perplexity,
        """
        return ind.transpose_(1, 2) if self.transpose else ind


class DVAEDecoder(nn.Module):
    def __init__(
        self,
        idim: int,
        odim: int,
        n_layer=12,
        bn_dim=64,
        hidden=256,
        kernel=7,
        dilation=2,
        up=False,
    ):
        super().__init__()
        self.up = up
        self.conv_in = nn.Sequential(
            nn.Conv1d(idim, bn_dim, 3, 1, 1),
            nn.GELU(),
            nn.Conv1d(bn_dim, hidden, 3, 1, 1),
        )
        self.decoder_block = nn.ModuleList(
            [
                ConvNeXtBlock(
                    hidden,
                    hidden * 4,
                    kernel,
                    dilation,
                )
                for _ in range(n_layer)
            ]
        )
        self.conv_out = nn.Conv1d(hidden, odim, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor, conditioning=None) -> torch.Tensor:
        # B, C, T
        y = self.conv_in(x)
        del x
        for f in self.decoder_block:
            y = f(y, conditioning)

        x = self.conv_out(y)
        del y
        return x


class MelSpectrogramFeatures(torch.nn.Module):
    def __init__(
        self,
        sample_rate=24000,
        n_fft=1024,
        hop_length=256,
        n_mels=100,
        padding: Literal["center", "same"] = "center",
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.device = device
        if padding not in ["center", "same"]:
            raise ValueError("Padding must be 'center' or 'same'.")
        self.padding = padding
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            center=padding == "center",
            power=1,
        )

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        return super().__call__(audio)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        audio = audio.to(self.device)
        mel: torch.Tensor = self.mel_spec(audio)
        features = torch.log(torch.clip(mel, min=1e-5))
        return features


class DVAE(nn.Module):
    def __init__(
        self,
        decoder_config: dict,
        encoder_config: Optional[dict] = None,
        vq_config: Optional[dict] = None,
        dim=512,
        coef: Optional[str] = None,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        coef = torch.rand(100)
        self.register_buffer("coef", coef.unsqueeze(0).unsqueeze_(2))

        if encoder_config is not None:
            self.downsample_conv = nn.Sequential(
                nn.Conv1d(100, dim, 3, 1, 1),
                nn.GELU(),
                nn.Conv1d(dim, dim, 4, 2, 1),
                nn.GELU(),
            )
            self.preprocessor_mel = MelSpectrogramFeatures(device=device)
            self.encoder: Optional[DVAEDecoder] = DVAEDecoder(**encoder_config)

        self.decoder = DVAEDecoder(**decoder_config)
        self.out_conv = nn.Conv1d(dim, 100, 3, 1, 1, bias=False)
        if vq_config is not None:
            self.vq_layer = GFSQ(**vq_config)
        else:
            self.vq_layer = None

    def __call__(
        self, inp: torch.Tensor, mode: Literal["encode", "decode"] = "decode"
    ) -> torch.Tensor:
        return super().__call__(inp, mode)

    @torch.inference_mode()
    def load_pretrained(self, filename: str, device: torch.device):
        state_dict_tensors = torch.load(filename)
        self.load_state_dict(state_dict_tensors)
        self.to(device)

    @torch.inference_mode()
    def forward(
        self, inp: torch.Tensor, mode: Literal["encode", "decode"] = "decode"
    ) -> torch.Tensor:
        if mode == "encode" and hasattr(self, "encoder") and self.vq_layer is not None:
            mel = self.preprocessor_mel(inp)
            x: torch.Tensor = self.downsample_conv(
                torch.div(mel, self.coef.view(100, 1).expand(mel.shape), out=mel),
            ).unsqueeze_(0)
            del mel
            x = self.encoder(x)
            ind = self.vq_layer(x)
            del x
            return ind

        if self.vq_layer is not None:
            vq_feats = self.vq_layer._embed(inp)
        else:
            vq_feats = inp

        vq_feats = (
            vq_feats.view(
                (vq_feats.size(0), 2, vq_feats.size(1) // 2, vq_feats.size(2)),
            )
            .permute(0, 2, 3, 1)
            .flatten(2)
        )

        dec_out = self.out_conv(
            self.decoder(
                x=vq_feats,
            ),
        )

        del vq_feats

        return torch.mul(dec_out, self.coef, out=dec_out)

    @torch.inference_mode()
    def sample_audio(self, wav: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if isinstance(wav, np.ndarray):
            wav = torch.from_numpy(wav)
        return self(wav, "encode").squeeze_(0)


@dataclass(repr=False, eq=False)
class DecoderConfig:
    idim: int = 384
    odim: int = 384
    hidden: int = 512
    n_layer: int = 12
    bn_dim: int = 128


@dataclass(repr=False, eq=False)
class VQConfig:
    dim: int = 1024
    levels: tuple = (5, 5, 5, 5)
    G: int = 2
    R: int = 2


@dataclass(repr=False, eq=False)
class DVAEConfig:
    encoder: DecoderConfig = DecoderConfig(
        idim=512,
        odim=1024,
        hidden=256,
        n_layer=12,
        bn_dim=128,
    )
    decoder: DecoderConfig = DecoderConfig(
        idim=512,
        odim=512,
        hidden=256,
        n_layer=12,
        bn_dim=128,
    )
    vq: VQConfig = VQConfig()


@dataclass(repr=False, eq=False)
class FeatureExtractorInitArgs:
    sample_rate: int = 24000
    n_fft: int = 1024
    hop_length: int = 256
    n_mels: int = 100
    padding: str = "center"


@dataclass(repr=False, eq=False)
class FeatureExtractor:
    class_path: str = "vocos.feature_extractors.MelSpectrogramFeatures"
    init_args: FeatureExtractorInitArgs = FeatureExtractorInitArgs()


@dataclass(repr=False, eq=False)
class BackboneInitArgs:
    input_channels: int = 100
    dim: int = 512
    intermediate_dim: int = 1536
    num_layers: int = 8


@dataclass(repr=False, eq=False)
class Backbone:
    class_path: str = "vocos.models.VocosBackbone"
    init_args: BackboneInitArgs = BackboneInitArgs()


@dataclass(repr=False, eq=False)
class FourierHeadInitArgs:
    dim: int = 512
    n_fft: int = 1024
    hop_length: int = 256
    padding: str = "center"


@dataclass(repr=False, eq=False)
class FourierHead:
    class_path: str = "vocos.heads.ISTFTHead"
    init_args: FourierHeadInitArgs = FourierHeadInitArgs()


@dataclass(repr=False, eq=False)
class VocosConfig:
    feature_extractor: FeatureExtractor = FeatureExtractor()
    backbone: Backbone = Backbone()
    head: FourierHead = FourierHead()


if __name__ == "__main__":
    import librosa
    import soundfile as sf

    # download model first, from:
    # https://huggingface.co/2Noise/ChatTTS/resolve/main/asset/DVAE_full.pt?download=true
    # https://huggingface.co/2Noise/ChatTTS/resolve/main/asset/Vocos.pt?download=true

    vocos_ckpt_path = "/mnt/data/user/tc_agi/xubokai/Vocos.pt"
    dvae_ckpt_path = "/mnt/data/user/tc_agi/xubokai/DVAE_full.pt"

    device = "cuda"
    vocos_config = VocosConfig()
    feature_extractor = instantiate_class(
        args=(), init=asdict(vocos_config.feature_extractor)
    )
    backbone = instantiate_class(args=(), init=asdict(vocos_config.backbone))
    head = instantiate_class(args=(), init=asdict(vocos_config.head))
    vocos = (
        Vocos(feature_extractor=feature_extractor, backbone=backbone, head=head)
        .to(device)
        .eval()
    )

    vocos.load_state_dict(torch.load(vocos_ckpt_path))

    dvae_config = DVAEConfig()

    dvae = DVAE(
        decoder_config=asdict(dvae_config.decoder),
        encoder_config=asdict(dvae_config.encoder),
        vq_config=asdict(dvae_config.vq),
        dim=dvae_config.decoder.idim,
        coef=None,
        device=device,
    )

    dvae.load_pretrained(dvae_ckpt_path, device)

    dvae = dvae.eval()

    from IPython import embed

    embed()

    audio_path = "/mnt/data/user/tc_agi/xubokai/icl_20.wav"
    waveform, _ = librosa.load(audio_path, sr=24000, mono=True)
    waveform = torch.tensor(waveform).to(device)
    codes = dvae(inp=waveform, mode="encode")

    reconstructed_mel = dvae(inp=codes, mode="decode")
    reconstructed_waveform = vocos.decode(reconstructed_mel).cpu().numpy()

    waveform_mono = reconstructed_waveform.squeeze()  # 形状变为 (24000,)
    # waveform_mono = waveform.flatten() # 效果相同

    # 如果希望采样率是 24000 Hz，写出 WAV 文件
    sf.write("output.wav", waveform_mono, samplerate=24000, subtype="PCM_16")
