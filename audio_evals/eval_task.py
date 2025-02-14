import json
import traceback
from concurrent.futures import as_completed, ThreadPoolExecutor
from functools import lru_cache
from typing import Dict, List, Tuple, Union

from tqdm import tqdm

from audio_evals.agg.base import AggPolicy
from audio_evals.base import ScoreUnit
from audio_evals.dataset.dataset import Dataset
from audio_evals.evaluator.base import Evaluator
from audio_evals.models.model import Model
from audio_evals.process.base import Process
from audio_evals.prompt.base import Prompt
from audio_evals.recorder import Recorder


def extract_score(s: str):
    d = json.loads(s)
    return d["score"]


class EvalTask:

    def __init__(
        self,
        dataset: Dataset,
        prompt: Prompt,
        predictor: Model,
        evaluator: Evaluator,
        post_process: List[Process],
        agg: AggPolicy,
        recorder: Recorder,
    ):
        self.dataset = dataset
        self.prompt = prompt
        self.predictor = predictor
        self.evaluator = evaluator
        self.post_process = post_process
        self.agg = agg
        self.recorder = recorder

    def _eval(
        self, idx, prompt: Union[str, List[Dict[str, str]]], reference: str, **kwargs
    ) -> Tuple[ScoreUnit, str]:
        self.recorder.add({"type": "prompt", "id": idx, "data": {"content": prompt}})
        if "eval_info" in kwargs and "inference" in kwargs["eval_info"]:
            output = kwargs["eval_info"]["inference"]["content"]
        else:
            output = self.predictor.inference(prompt)
        self.recorder.add({"type": "inference", "id": idx, "data": {"content": output}})

        if "eval_info" in kwargs and "post_process" in kwargs["eval_info"]:
            output = kwargs["eval_info"]["post_process"]["content"]
        else:
            for p in self.post_process:
                output = p(output)
        self.recorder.add(
            {"type": "post_process", "id": idx, "data": {"content": output}}
        )

        if "eval_info" in kwargs and "eval" in kwargs["eval_info"]:
            score = kwargs["eval_info"]["eval"]
        else:
            score = self.evaluator(output, reference, **kwargs)
        self.recorder.add({"type": "eval", "id": idx, "data": score})
        return score, output

    def _run(self, i, doc):
        """单个任务处理逻辑"""
        real_prompt = self.prompt.load(**doc)
        try:
            score, ans = self._eval(
                i, real_prompt, doc.get(self.dataset.ref_col, ""), **doc
            )
            return i, score, ans, 0
        except Exception:
            error_traceback = traceback.format_exc()
            self.recorder.add(
                {"type": "error", "id": i, "data": {"info": error_traceback}}
            )
            print(error_traceback)
            return i, None, None, 1

    @lru_cache(maxsize=None)
    def run(
        self, limit=None, rand_size=None, max_workers=1
    ) -> Tuple[ScoreUnit, List[ScoreUnit], List[str]]:
        """
        eval
        :param :
        :return:
        """
        quiz = self.dataset.load(limit)
        if limit:
            quiz = quiz[:limit]
        if rand_size:
            import random

            quiz = random.sample(quiz, rand_size)

        res = [None] * len(quiz)
        answers = [None] * len(quiz)
        error_count = 0

        # 使用进程池并控制最大并发量
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = [
                executor.submit(self._run, i, doc) for i, doc in enumerate(quiz)
            ]
            for future in tqdm(as_completed(future_to_index), total=len(quiz)):
                index, score, ans, has_error = future.result()
                error_count += has_error
                if score is not None and ans is not None:
                    res[index] = score
                    answers[index] = ans

        res, answers = [item for item in res if item is not None], [
            item for item in answers if item is not None
        ]
        final_res = self.agg(res)
        final_res["fail_rate(%d)"] = error_count / len(quiz) * 100
        return final_res, res, answers
