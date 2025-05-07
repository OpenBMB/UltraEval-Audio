from .base import Evaluator
from typing import Dict, List
import json


class BBH(Evaluator):
    def __init__(self, ignore_case: bool = True):
        self.ignore_case = ignore_case

    def _extract_answer(self, response: str) -> str:
        response = response.lower() if self.ignore_case else response

        # 尝试从 JSON 格式中提取答案
        try:
            data = json.loads(response)
            if isinstance(data, dict) and "answer" in data:
                return data["answer"]
        except:
            pass

        # 尝试从文本中提取答案
        for line in response.split("\n"):
            line = line.strip()
            if line.startswith("answer:") or line.startswith("Answer:"):
                return line.split(":", 1)[1].strip()
            if line.startswith("the answer is") or line.startswith("The answer is"):
                return line.split("is", 1)[1].strip()

        return None

    def _eval(self, pred: str, label: str, **kwargs) -> Dict[str, any]:
        pred = str(pred)
        label = str(label)

        if self.ignore_case:
            pred = pred.lower()
            label = label.lower()

        extracted_answer = self._extract_answer(pred)
        if extracted_answer is None:
            return {"match": 0, "pred": pred, "ref": label, "fail": 1}

        return {
            "match": 1 if extracted_answer == label else 0,
            "pred": extracted_answer,
            "ref": label,
            "fail": 0,
        }
