from typing import Dict
import os
from audio_evals.evaluator.base import Evaluator
import zhconv
from jiwer import compute_measures
from zhon.hanzi import punctuation
import string
import logging

logger = logging.getLogger(__name__)

punctuation_all = punctuation + string.punctuation


def process_one(hypo, truth, lang):
    raw_truth = truth
    raw_hypo = hypo

    for x in punctuation_all:
        if x == "'":
            continue
        truth = truth.replace(x, "")
        hypo = hypo.replace(x, "")

    truth = truth.replace("  ", " ")
    hypo = hypo.replace("  ", " ")

    # yue: cantonese, th: thai
    if lang in ["zh", "ja", "yue", "th", "ko"]:
        truth = " ".join([x for x in truth if x.strip()])
        hypo = " ".join([x for x in hypo if x.strip()])
    # Word-based languages (WER)
    else:
        truth = truth.lower()
        hypo = hypo.lower()

    try:
        measures = compute_measures(truth, hypo)
        wer = measures["wer"]
    except Exception as e:
        logger.error(f"Error computing measures: {e}. truth: '{truth}', hypo: '{hypo}'")
        raise e

    return wer


class SeedTTSEvalASRWER(Evaluator):
    def __init__(self, model_name, prompt_name, lang):
        from audio_evals.registry import registry

        self.prompt = registry.get_prompt(prompt_name)
        self.model = registry.get_model(model_name)
        self.lang = lang

    def _eval(self, pred, label, **kwargs) -> Dict[str, any]:
        pred = str(pred)
        label_text = kwargs["text"]
        assert os.path.exists(pred), "must be a valid audio file, but got {}".format(
            pred
        )

        real_prompt = self.prompt.load(WavPath=pred)

        # Pass language to model for non-Chinese languages or if specified
        # Whisper model expects language in generate_kwargs
        inf_kwargs = {}
        if self.lang != "zh":
            inf_kwargs["generate_kwargs"] = {
                "language": kwargs.get("language", self.lang)
            }

        transcription = self.model.inference(real_prompt, **inf_kwargs)

        if self.lang == "zh" or kwargs.get("language") == "chinese":
            transcription = zhconv.convert(transcription, "zh-cn")

        res = {"wer%": process_one(transcription, label_text, self.lang) * 100}
        res.update(
            {
                "transcription": transcription,
                "label_text": label_text,
                "pred": pred,
                "ref": label,
            }
        )
        return res
