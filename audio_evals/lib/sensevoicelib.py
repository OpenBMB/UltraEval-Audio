from typing import List, Dict
import os
from asr_decoder import CTCDecoder
from online_fbank import OnlineFbank
import numpy as np
import onnxruntime
import sentencepiece as spm
from asr_decoder.context_graph import ContextGraph
from asr_decoder.prefix_score import PrefixScore
from asr_decoder.utils import log_add
from collections import defaultdict
import librosa
import sys


def load_cmvn(cmvn_file):
    with open(cmvn_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    means_list = []
    vars_list = []
    for i in range(len(lines)):
        line_item = lines[i].split()
        if line_item[0] == "<AddShift>":
            line_item = lines[i + 1].split()
            if line_item[0] == "<LearnRateCoef>":
                add_shift_line = line_item[3 : (len(line_item) - 1)]
                means_list = list(add_shift_line)
                continue
        elif line_item[0] == "<Rescale>":
            line_item = lines[i + 1].split()
            if line_item[0] == "<LearnRateCoef>":
                rescale_line = line_item[3 : (len(line_item) - 1)]
                vars_list = list(rescale_line)
                continue
    means = np.array(means_list).astype(np.float32)
    vars = np.array(vars_list).astype(np.float32)
    cmvn = np.array([means, vars])
    return cmvn


class CTCDecoder:
    def __init__(
        self,
        contexts: List[str] = None,
        symbol_table: Dict[str, int] = None,
        bpe_model: str = None,
        context_score: float = 6.0,
        blank_id: int = 0,
    ):
        self.context_graph = None
        if contexts is not None:
            self.context_graph = ContextGraph(contexts, symbol_table, bpe_model)
        self.blank_id = blank_id
        self.cur_t = 0
        self.cur_hyps = []
        self.reset()

    def reset(self):
        self.cur_t = 0
        context_root = None if self.context_graph is None else self.context_graph.root
        self.cur_hyps = [
            (tuple(), PrefixScore(s=0.0, v_s=0.0, context_state=context_root))
        ]

    def copy_context(self, prefix_score, next_score):
        # perfix not changed, copy the context from prefix
        if self.context_graph is not None and not next_score.has_context:
            next_score.context_score = prefix_score.context_score
            next_score.context_state = prefix_score.context_state
            next_score.has_context = True

    def update_context(self, prefix_score, next_score, word_id):
        if self.context_graph is not None and not next_score.has_context:
            context_score, context_state = self.context_graph.forward_one_step(
                prefix_score.context_state, word_id
            )
            next_score.context_score = prefix_score.context_score + context_score
            next_score.context_state = context_state
            next_score.has_context = True

    def backoff_context(self):
        if self.context_graph is not None:
            # We should backoff the context score/state when the context is
            # not fully matched at the last time.
            for i, hyp in enumerate(self.cur_hyps):
                score, new_state = self.context_graph.finalize(hyp[1].context_state)
                self.cur_hyps[i][1].context_score = score
                self.cur_hyps[i][1].context_state = new_state

    def ctc_greedy_search(self, ctc_probs: np.ndarray, is_last: bool = False):
        results = self.ctc_prefix_beam_search(ctc_probs, 1, is_last)
        return {"tokens": results["tokens"][0], "times": results["times"][0]}

    def ctc_prefix_beam_search(
        self, ctc_probs: np.ndarray, beam_size: int, is_last: bool = False
    ):
        for logp in ctc_probs:
            self.cur_t += 1
            # key: prefix, value: PrefixScore
            next_hyps = defaultdict(lambda: PrefixScore())
            # 1. First beam prune: select topk best

            indices = np.argpartition(logp, -beam_size)[-beam_size:]
            logp = logp[indices]

            for prob, u in zip(logp.tolist(), indices.tolist()):
                for prefix, prefix_score in self.cur_hyps:
                    last = prefix[-1] if len(prefix) > 0 else None
                    if u == self.blank_id:  # blank
                        next_score = next_hyps[prefix]
                        next_score.s = log_add(
                            next_score.s, prefix_score.score() + prob
                        )
                        next_score.v_s = prefix_score.viterbi_score() + prob
                        next_score.times_s = prefix_score.times().copy()
                        # perfix not changed, copy the context from prefix
                        self.copy_context(prefix_score, next_score)
                    elif u == last:
                        # Update *uu -> *u;
                        next_score1 = next_hyps[prefix]
                        next_score1.ns = log_add(next_score1.ns, prefix_score.ns + prob)
                        if next_score1.v_ns < prefix_score.v_ns + prob:
                            next_score1.v_ns = prefix_score.v_ns + prob
                            if next_score1.cur_token_prob < prob:
                                next_score1.cur_token_prob = prob
                                next_score1.times_ns = prefix_score.times_ns.copy()
                                next_score1.times_ns[-1] = self.cur_t
                        self.copy_context(prefix_score, next_score1)
                        # Update *u-u -> *uu, - is for blank
                        n_prefix = prefix + (u,)
                        next_score2 = next_hyps[n_prefix]
                        next_score2.ns = log_add(next_score2.ns, prefix_score.s + prob)
                        if next_score2.v_ns < prefix_score.v_s + prob:
                            next_score2.v_ns = prefix_score.v_s + prob
                            next_score2.cur_token_prob = prob
                            next_score2.times_ns = prefix_score.times_s.copy()
                            next_score2.times_ns.append(self.cur_t)
                        self.update_context(prefix_score, next_score2, u)
                    else:
                        n_prefix = prefix + (u,)
                        next_score = next_hyps[n_prefix]
                        next_score.ns = log_add(
                            next_score.ns, prefix_score.score() + prob
                        )
                        if next_score.v_ns < prefix_score.viterbi_score() + prob:
                            next_score.v_ns = prefix_score.viterbi_score() + prob
                            next_score.cur_token_prob = prob
                            next_score.times_ns = prefix_score.times().copy()
                            next_score.times_ns.append(self.cur_t)
                        self.update_context(prefix_score, next_score, u)

            # 2. Second beam prune
            next_hyps = sorted(
                next_hyps.items(), key=lambda x: x[1].total_score(), reverse=True
            )
            self.cur_hyps = next_hyps[:beam_size]

        if is_last:
            self.backoff_context()

        return {
            "tokens": [list(y[0]) for y in self.cur_hyps],
            "times": [y[1].times() for y in self.cur_hyps],
        }


class StreamingSenseVoice:
    def __init__(
        self,
        model_path: str,
        num_cpus: int = 4,
    ):
        chunk_size = 16
        padding = 8

        sess_opt = onnxruntime.SessionOptions()
        sess_opt.intra_op_num_threads = num_cpus
        self.sess = onnxruntime.InferenceSession(
            os.path.join(model_path, "sensevoicesmall.onnx"),
            sess_opt,
            providers=["CPUExecutionProvider"],
        )

        # features
        cmvn = load_cmvn(os.path.join(model_path, "am.mvn"))
        self.neg_mean, self.inv_stddev = cmvn[0, :], cmvn[1, :]
        self.fbank = OnlineFbank(window_type="hamming")
        # decoder
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.Load(
            os.path.join(model_path, "chn_jpn_yue_eng_ko_spectok.bpe.model")
        )

        self.lid_dict = {
            "auto": 0,
            "zh": 3,
            "en": 4,
            "yue": 7,
            "ja": 11,
            "ko": 12,
            "nospeech": 13,
        }
        self.textnorm_dict = {"withitn": 14, "woitn": 15}

        symbol_table = {}
        for i in range(self.tokenizer.GetPieceSize()):
            symbol_table[self.tokenizer.DecodeIds(i)] = i

        self.beam_size = 1
        self.decoder = CTCDecoder()

        self.chunk_size = chunk_size
        self.padding = padding
        self.cur_idx = -1
        self.caches_shape = (chunk_size + 2 * padding, 560)
        self.caches = np.zeros(self.caches_shape, dtype=np.float32)

    def reset(self):
        self.cur_idx = -1
        self.decoder.reset()
        self.fbank = OnlineFbank(window_type="hamming")
        self.caches = np.zeros(self.caches_shape, dtype=np.float32)

    def get_size(self):
        effective_size = self.cur_idx + 1 - self.padding
        if effective_size <= 0:
            return 0
        return effective_size % self.chunk_size or self.chunk_size

    def inference(self, speech, query):
        speech = speech[None, :, :]
        speech_lengths = np.array([speech.shape[1]], dtype=np.int64)
        return self.sess.run(
            ["logits"],
            {
                "speech": speech,
                "speech_lengths": speech_lengths,
                "query": query,
            },
        )[0]

    def decode(self, times, tokens):
        times_ms = []
        for step, token in zip(times, tokens):
            if len(self.tokenizer.DecodeIds(token).strip()) == 0:
                continue
            times_ms.append(step * 60)
        return times_ms, self.tokenizer.DecodeIds(tokens)

    def streaming_inference(
        self, audio, is_last, language: str = "auto", textnorm: bool = False
    ):
        self.fbank.accept_waveform(audio, is_last)
        features = self.fbank.get_lfr_frames(
            neg_mean=self.neg_mean, inv_stddev=self.inv_stddev
        )
        unbind_features = [features[i] for i in range(features.shape[0])]
        query = np.array(
            [
                self.lid_dict[language],
                1,
                2,
                self.textnorm_dict["withitn" if textnorm else "woitn"],
            ],
            dtype=np.int64,
        )

        for feature in unbind_features:
            self.caches = np.roll(self.caches, -1, axis=0)
            self.caches[-1, :] = feature
            self.cur_idx += 1
            cur_size = self.get_size()
            if cur_size != self.chunk_size and not is_last:
                continue
            probs = self.inference(self.caches, query)[self.padding :]
            if cur_size != self.chunk_size:
                probs = probs[self.chunk_size - cur_size :]
            if not is_last:
                probs = probs[: self.chunk_size]
            if self.beam_size > 1:
                res = self.decoder.ctc_prefix_beam_search(
                    probs, beam_size=self.beam_size, is_last=is_last
                )
                times_ms, text = self.decode(res["times"][0], res["tokens"][0])
            else:
                res = self.decoder.ctc_greedy_search(probs, is_last=is_last)
                times_ms, text = self.decode(res["times"], res["tokens"])
            yield {"timestamps": times_ms, "text": text}


def main():
    if len(sys.argv) < 2:
        print("need argv: python3 test.wav")
        return -1
    file_name = sys.argv[1]
    model = StreamingSenseVoice("model")
    wav, _ = librosa.load(file_name, sr=16000)
    step = int(0.1 * 16_000)

    import time

    st = time.time()
    for i in range(0, len(wav), step):
        is_last = i + step >= len(wav)
        for res in model.streaming_inference(wav[i : i + step], is_last):
            print(time.time() - st, ":", res["text"])


if __name__ == "__main__":
    main()
