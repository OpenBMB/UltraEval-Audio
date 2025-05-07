from .base import Evaluator
import numpy as np
from typing import Dict, List
from qa_metrics.pedant import PEDANT


def majority_vote(scores: List[str]) -> bool:
    scores = [item.lower() for item in scores]
    final_answer = max(set(scores), key=scores.count)
    return True if final_answer == "yes" else False


class QAEval(Evaluator):
    def __init__(self):
        self.pedant = PEDANT()

    def _eval(self, pred: str, label: str, **kwargs) -> Dict[str, any]:
        pred = str(pred)
        label = str(label)

        # 使用 PEDANT 进行评测
        panda_score = self.pedant.evaluate(
            [label.lower()], pred.lower(), kwargs.get("prompt", "").lower()
        )

        # 使用多数投票机制
        gpt_score = majority_vote([pred])

        return {
            "panda_score": panda_score * 100,
            "gpt_score": gpt_score * 100,
            "pred": pred,
            "ref": label,
        }
