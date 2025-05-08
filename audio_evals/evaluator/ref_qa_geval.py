import ast
import os.path
import re
from copy import deepcopy
from typing import Dict
import yaml
import json
from audio_evals.evaluator.base import Evaluator

path = os.path.dirname(__file__)
prompt = open(os.path.join(path, "ref_qa_geval.txt"), "r").read()


class RefQAGEval(Evaluator):
    def __init__(self, model_name: str):
        self.model_name = model_name

    def _eval(self, pred, label, **kwargs) -> Dict[str, any]:
        from audio_evals.registry import registry

        model = registry.get_model(self.model_name)

        p = deepcopy(prompt)
        for k, v in {
            "question": kwargs["question"],
            "prediction": pred,
            "answer": label,
        }.items():
            p = p.replace(f"{{{k}}}", v)

        res = model.inference(p, temperature=0)
        score = res.strip().split("\n")[-1]
        match = None
        if "yes" in score.lower():
            match = 1
        elif "no" in score.lower():
            match = 0
        else:
            raise ValueError(
                "the eval output is illeagal, should contain yes or no, but got {}".format(
                    res
                )
            )

        return {
            "acc": match,
            "pred": pred,
            "ref": label,
        }
