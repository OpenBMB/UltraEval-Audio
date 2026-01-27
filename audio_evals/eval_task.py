import json
import traceback
import threading
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
from audio_evals.utils import logger, merge_data4view


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
        self._evaluator_lock = threading.Lock()  # 确保 evaluator 不并发执行

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
            with self._evaluator_lock:  # 确保 evaluator 串行执行
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

    def _inference_only(self, i, doc):
        """仅执行推理，不进行评测"""
        real_prompt = self.prompt.load(**doc)
        try:
            self.recorder.add({"type": "prompt", "id": i, "data": {"content": real_prompt}})
            
            if "eval_info" in doc and "inference" in doc["eval_info"]:
                output = doc["eval_info"]["inference"]["content"]
            else:
                output = self.predictor.inference(real_prompt)
            self.recorder.add({"type": "inference", "id": i, "data": {"content": output}})
            
            if "eval_info" in doc and "post_process" in doc["eval_info"]:
                output = doc["eval_info"]["post_process"]["content"]
            else:
                for p in self.post_process:
                    output = p(output)
            self.recorder.add({"type": "post_process", "id": i, "data": {"content": output}})
            
            return i, output, doc, 0
        except Exception:
            error_traceback = traceback.format_exc()
            self.recorder.add({"type": "error", "id": i, "data": {"info": error_traceback}})
            print(error_traceback)
            return i, None, doc, 1

    def _evaluate_only(self, i, output, doc):
        """仅执行评测，假设推理已完成"""
        try:
            reference = doc.get(self.dataset.ref_col, "")
            score = self.evaluator(output, reference, **doc)
            self.recorder.add({"type": "eval", "id": i, "data": score})
            return i, score, output, 0
        except Exception:
            error_traceback = traceback.format_exc()
            self.recorder.add({"type": "error", "id": i, "data": {"info": error_traceback}})
            print(error_traceback)
            return i, None, output, 1

    def _release_predictor(self):
        """释放推理模型占用的 GPU 显存"""
        try:
            # 尝试调用模型的释放方法（如果有的话）
            if hasattr(self.predictor, 'release') and callable(self.predictor.release):
                self.predictor.release()
                # 删除模型引用
                del self.predictor
                self.predictor = None
                print("Predictor released successfully, GPU memory freed.")
            else:
                predictor_type = type(self.predictor).__name__
                print(f"Predictor ({predictor_type}) does not have a release method, skipping GPU memory release.")
        except Exception as e:
            print(f"Warning: Failed to release GPU memory: {e}")

    def run_two_phase(
        self, limit=None, rand_size=None, max_workers=1
    ) -> Tuple[ScoreUnit, List[ScoreUnit], List[str]]:
        """
        两阶段执行：先并发推理，后串行评测
        :param limit: 限制数据条数
        :param rand_size: 随机采样数量
        :param max_workers: 推理阶段的并发数
        :return: 聚合结果, 各条评分, 各条输出
        """
        quiz = self.dataset.load(limit)
        if limit:
            quiz = quiz[:limit]
        if rand_size:
            import random
            quiz = random.sample(quiz, rand_size)

        # 第一阶段：并发推理
        print("Phase 1: Running inference...")
        inference_results = [None] * len(quiz)
        inference_docs = [None] * len(quiz)
        inference_error_count = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = [
                executor.submit(self._inference_only, i, doc) for i, doc in enumerate(quiz)
            ]
            for future in tqdm(as_completed(future_to_index), total=len(quiz), desc="Inference"):
                index, output, doc, has_error = future.result()
                inference_error_count += has_error
                if output is not None:
                    inference_results[index] = output
                    inference_docs[index] = doc

        # 释放模型显存，评测阶段不需要 GPU
        print("Releasing model GPU memory...")
        self._release_predictor()

        # 第二阶段：串行评测
        print("Phase 2: Running evaluation...")
        res = [None] * len(quiz)
        answers = [None] * len(quiz)
        eval_error_count = 0

        for i in tqdm(range(len(quiz)), desc="Evaluation"):
            if inference_results[i] is not None:
                index, score, output, has_error = self._evaluate_only(
                    i, inference_results[i], inference_docs[i]
                )
                eval_error_count += has_error
                if score is not None:
                    res[index] = score
                    answers[index] = output

        res, answers = [item for item in res if item is not None], [
            item for item in answers if item is not None
        ]
        merge_data4view(
            quiz, self.recorder.name, self.recorder.name.replace(".jsonl", ".xlsx")
        )
        final_res = self.agg(res)
        total_error = inference_error_count + eval_error_count
        final_res["fail_rate(%d)"] = total_error / len(quiz) * 100
        final_res["inference_fail_count"] = inference_error_count
        final_res["eval_fail_count"] = eval_error_count
        return final_res, res, answers

    @lru_cache(maxsize=None)
    def run(
        self, limit=None, rand_size=None, max_workers=1, two_phase=False
    ) -> Tuple[ScoreUnit, List[ScoreUnit], List[str]]:
        """
        eval
        :param limit: 限制数据条数
        :param rand_size: 随机采样数量
        :param max_workers: 并发数
        :param two_phase: 是否使用两阶段模式（先推理后评测）
        :return: 聚合结果, 各条评分, 各条输出
        """
        if two_phase:
            return self.run_two_phase(limit, rand_size, max_workers)
            
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
        merge_data4view(
            quiz, self.recorder.name, self.recorder.name.replace(".jsonl", ".xlsx")
        )
        final_res = self.agg(res)
        final_res["fail_rate(%d)"] = error_count / len(quiz) * 100
        return final_res, res, answers
