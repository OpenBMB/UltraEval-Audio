import json
from typing import Dict

from audio_evals.evaluator.base import Evaluator


class DNSMOS(Evaluator):
    def __init__(self, model_name: str = "DNSMOS"):
        from audio_evals.registry import registry

        self.model = registry.get_model(model_name)

    def _eval(self, pred, label, **kwargs) -> Dict[str, any]:
        pred = {"audio": str(pred)}
        res = self.model.inference(pred)
        res = json.loads(res)
        res["pred"] = pred
        res["ref"] = label
        return res
