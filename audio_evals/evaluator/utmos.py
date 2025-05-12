from typing import Dict

from audio_evals.evaluator.base import Evaluator


class UTMOS(Evaluator):
    def __init__(self, model_name: str = "utmos-en"):
        from audio_evals.registry import registry

        self.model = registry.get_model(model_name)

    def _eval(self, pred, label, **kwargs) -> Dict[str, any]:
        pred = str(pred)
        return {
            "utmos": self.model.inference(pred),
            "pred": pred,
            "ref": label,
        }
