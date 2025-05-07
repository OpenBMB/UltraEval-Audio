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

    def _eval(self, pred, label, **kwargs) -> Dict[str, any]:
        pred = str(pred)
        assert os.path.exists(label), f"Label file {label} does not exist"
        return {
            "simo": self.model.inference({"audios": [pred, label]}),
            "pred": pred,
            "ref": label,
        }
