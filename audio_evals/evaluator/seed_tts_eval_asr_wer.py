from typing import Dict
import os
from audio_evals.evaluator.base import Evaluator
import zhconv


class SeedTTSEvalASRWER(Evaluator):
    def __init__(self, model_name, prompt_name, lang):
        from audio_evals.registry import registry

        self.prompt = registry.get_prompt(prompt_name)
        self.model = registry.get_model(model_name)
        self.lang = lang
        if lang == "zh":
            self.e = registry.get_evaluator("naive-wer-zh")
        elif lang == "en":
            self.e = registry.get_evaluator("naive-wer-en")
        else:
            raise ValueError(f"Unsupported language: {lang}")

    def _eval(self, pred, label, **kwargs) -> Dict[str, any]:
        pred = str(pred)
        label_text = kwargs["text"]
        assert os.path.exists(pred), "must be a valid audio file, but got {}".format(
            pred
        )
        real_prompt = self.prompt.load(WavPath=pred)
        transcription = self.model.inference(real_prompt)
        if self.lang == "zh":
            transcription = zhconv.convert(transcription, "zh-cn")

        res = self.e(transcription, label_text)
        res.update(
            {
                "transcription": transcription,
                "label_text": label_text,
                "pred": pred,
                "ref": label,
            }
        )
        return res
