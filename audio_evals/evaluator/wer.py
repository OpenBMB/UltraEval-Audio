from jiwer import cer

from audio_evals.evaluator.base import Evaluator
from audio_evals.lib.wer import compute_wer


class WER(Evaluator):
    def __init__(self, ignore_case: bool = False):
        self.ignore_case = ignore_case

    def _eval(self, pred: str, label: str, **kwargs):
        pred, label = str(pred), str(label)
        if self.ignore_case:
            pred, label = pred.lower(), label.lower()
        return {"wer%": compute_wer([label], [pred]) * 100}


class CER(Evaluator):
    def __init__(self, ignore_case: bool = False):
        self.ignore_case = ignore_case

    def _eval(self, pred: str, label: str, **kwargs):
        pred, label = str(pred), str(label)
        if self.ignore_case:
            pred, label = pred.lower(), label.lower()
        return {"cer%": cer(label, pred) * 100}
