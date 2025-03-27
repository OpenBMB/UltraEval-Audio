from typing import Dict

from audio_evals.evaluator.base import Evaluator


class DictEM(Evaluator):

    def _eval(self, pred, label, **kwargs) -> Dict[str, any]:
        assert isinstance(label, dict), "label must be dictionaries, but {}".format(
            type(label)
        )
        return {"match": 1 if pred == label else 0, "pred": pred, "ref": label}
