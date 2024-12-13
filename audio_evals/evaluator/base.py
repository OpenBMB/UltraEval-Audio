from abc import ABC, abstractmethod
from typing import Dict


class Evaluator(ABC):

    @abstractmethod
    def _eval(self, pred, label, **kwargs) -> Dict[str, any]:
        raise NotImplementedError()

    def __call__(self, pred, ref, **kwargs) -> Dict[str, any]:
        res = {"pred": pred, "ref": ref}
        eval_kwargs = {k: v for k, v in kwargs.items() if k not in ["pred", "label"]}
        res.update(self._eval(pred, ref, **eval_kwargs))
        return res


class Dump(Evaluator):

    def _eval(self, pred, label, **kwargs):
        return {}


class EM(Evaluator):

    def _eval(self, pred, label, **kwargs) -> Dict[str, any]:
        if type(label) in [int, float]:
            pred, label = float(pred), float(label)
        elif isinstance(label, str):
            pred, label = str(pred).strip(), label.strip()

        return {"match": 1 if pred == label else 0, "pred": pred, "ref": label}


class ExistMatch(Evaluator):

    def _eval(self, pred, label, **kwargs) -> Dict[str, any]:
        if isinstance(label, list):
            for item in label:
                ans = self._eval(pred, item, **kwargs)
                if ans["match"] == 1:
                    return ans
            return {"match": 0, "pred": pred, "ref": label}

        if type(label) in [int, float]:
            pred, label = float(pred), float(label)
        elif isinstance(label, str):
            pred, label = str(pred).strip().lower(), label.strip().lower()

        match = 0
        if label in pred:
            match = 1

        return {"match": match, "pred": label if match else pred, "ref": label}


class PrefixMatch(Evaluator):

    def __init__(self, ignore_case: bool = True):
        self.ignore_case = ignore_case

    def _eval(self, pred, label, **kwargs) -> Dict[str, any]:
        if self.ignore_case:
            pred = pred.lower().strip()
            label = str(label).lower().strip()
        n = len(label)
        return {
            "match": 1 if pred[:n] == label else 0,
            "pred": pred[:n],
            "ref": label,
        }
