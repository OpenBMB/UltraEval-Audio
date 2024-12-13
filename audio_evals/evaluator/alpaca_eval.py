import ast
import os.path
import re
from copy import deepcopy
from typing import Dict
import yaml
import json
from audio_evals.evaluator.base import Evaluator

path = os.path.dirname(__file__)
prompt = open(os.path.join(path, "alpaca_eval.txt"), "r").read()


class AlpacaEvaluator(Evaluator):
    def __init__(self, model_name: str):
        self.model_name = model_name

    def _eval(self, pred, label, **kwargs) -> Dict[str, any]:
        from audio_evals.registry import registry

        model = registry.get_model(self.model_name)

        p = deepcopy(prompt)
        for k, v in {"instruction": kwargs["instruction"],
                     "output_1": pred,
                     "output_2": label}.items():
            p = p.replace(f"{{{k}}}", v)

        # with open("/Users/a1/project/alpaca_eval-main/src/alpaca_eval/evaluators_configs/alpaca_eval_cot_gpt4_turbo_fn/configs.yaml", "r", encoding="utf-8") as f:
        #     d = yaml.safe_load(f.read())
        res = model.inference(p, temperature=0, maxTokens=100)

        # res_d = re.search(r"```json(.*?)```", res, re.DOTALL)
        # if res_d:
        #     d = json.loads(res_d.group(1))
        if res.startswith("```python"):
            res = res[9:-3].strip()
        elif res.startswith("```"):
            res = res[3:-3].strip()
        try:
            res = ast.literal_eval(res)
            if isinstance(res, dict):
                for k in res:
                    res = res[k]
                    break
            return {
                "acc": 1 if res[0]["model"] == "model_1" else 0,
                "pred": pred,
                "ref": label,
            }
        except Exception as e:
            print(f"output is {res}\nError: {e}")
            raise e


class ChatbotEvaluator(Evaluator):
    def __init__(self, model_name: str):
        self.model_name = model_name

    def _eval(self, pred, label, **kwargs) -> Dict[str, any]:
        from audio_evals.registry import registry

        model = registry.get_model(self.model_name)
        prompt = registry.get_prompt("chatbot-eval")

        p = prompt.load(instruction=kwargs["instruction"], response=pred)
        res = model.inference(p, temperature=0, maxTokens=2048)

        # res_d = re.search(r"```json(.*?)```", res, re.DOTALL)
        d = re.search(r'\[\[(\d+)\]\]', res)
        return {
            "geval": int(d.group(1)),
            "pred": pred,
            "ref": label,
        }