import os
from typing import Dict

from audio_evals.evaluator.base import Evaluator


class Simo(Evaluator):
    def __init__(
        self,
        model_name: str = "wavlm_large",
    ):
        from audio_evals.registry import registry

        self.model = registry.get_model(model_name)

    def _eval(self, pred, label, WavPath, **kwargs) -> Dict[str, any]:
        pred = str(pred)
        if not os.path.exists(label) and os.path.exists(WavPath):
            label = WavPath
        return {
            "simo": self.model.inference({"audios": [pred, label]}),
            "pred": pred,
            "ref": label,
        }
