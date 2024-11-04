from typing import Union, List

from audio_evals.evaluator.base import Evaluator
from audio_evals.lib.coco import compute_caption


class Coco(Evaluator):

    def _eval(self, pred: str, label: Union[str, List[str]], **kwargs):
        pred = str(pred)
        if isinstance(label, str):
            label = [label]
        return compute_caption([label], [pred])
