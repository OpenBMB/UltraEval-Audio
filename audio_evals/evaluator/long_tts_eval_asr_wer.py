from typing import Dict
import os
from audio_evals.evaluator.base import Evaluator
from jiwer import compute_measures
from zhon.hanzi import punctuation
import re
from num2words import num2words
import string


# https://github.com/dvlab-research/MGM-Omni/blob/main/mgm/eval/long_tts_eval/evaluation.py


def calc_wer(hypo, truth):
    measures = compute_measures(truth, hypo)
    ref_list = truth.split(" ")
    wer = measures["wer"]
    return wer, len(ref_list)


def normalize_text(text, language="en"):
    try:
        text = re.sub(
            r"\d+",
            lambda m: f" {num2words(int(m.group(0)), lang=('zh' if language == 'zh' else 'en'))} ",
            text,
        )
    except:
        text = text
    punctuation_all = punctuation + string.punctuation
    pattern = f"[{re.escape(punctuation_all)}]"
    text = re.sub(pattern, " ", text.lower())
    text = " ".join(text.split())
    if language == "en":
        text = " ".join(text.split())
    else:
        text = re.sub(r"(?<=[^\s])([\u4e00-\u9fff])", r" \1", text)
        text = re.sub(r"([\u4e00-\u9fff])(?=[^\s])", r"\1 ", text)
    return text


class LongTTSEvalASRWER(Evaluator):
    def __init__(self, model_name, prompt_name, lang):
        from audio_evals.registry import registry

        self.prompt = registry.get_prompt(prompt_name)
        self.model = registry.get_model(model_name)
        assert lang in ["zh", "en"]
        self.lang = lang

    def _eval(self, pred, label, **kwargs) -> Dict[str, any]:
        pred = str(pred)
        label_text, label_text2 = kwargs["text"], kwargs["text_norm"]
        assert os.path.exists(pred), "must be a valid audio file, but got {}".format(
            pred
        )
        real_prompt = self.prompt.load(WavPath=pred)
        raw_transcription = self.model.inference(real_prompt)

        transcription = normalize_text(raw_transcription, self.lang)

        wer1, ref_len1 = calc_wer(transcription, normalize_text(label_text, self.lang))
        wer2, ref_len2 = calc_wer(transcription, normalize_text(label_text2, self.lang))

        res = {"wer": wer1, "word_count": ref_len1}
        if wer2 < wer1:
            res["wer"] = wer2
            res["word_count"] = ref_len2
        res.update(
            {
                "transcription": raw_transcription,
                "label_text": label_text,
                "pred": pred,
                "ref": label,
            }
        )
        return res
