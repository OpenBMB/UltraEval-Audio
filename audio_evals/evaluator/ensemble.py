from typing import Dict, List

from audio_evals.evaluator.base import Evaluator


class Ensemble(Evaluator):
    def __init__(self, components: List[str]):
        from audio_evals.registry import registry

        self.es = []
        for item in components:
            e = registry.get_evaluator(item)
            if e is None:
                raise ValueError(f"Invalid component: {item}")
            self.es.append(e)

    def _eval(self, pred, label, **kwargs) -> Dict[str, any]:
        res = {}
        for e in self.es:
            res.update(e(pred, label, **kwargs))
        return res
