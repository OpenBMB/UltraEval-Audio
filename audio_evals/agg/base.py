from abc import ABC, abstractmethod
from typing import Dict, List

import pandas as pd
import sacrebleu
from jiwer import cer, wer
from sklearn.metrics import accuracy_score

from audio_evals.lib.coco import compute_caption
from audio_evals.lib.wer import compute_wer


class AggPolicy(ABC):
    def __init__(self, need_score_col: List[str] = None):
        self.need_score_col = need_score_col
        if need_score_col is None:
            self.need_score_col = []

    @abstractmethod
    def _agg(self, score_detail: List[Dict[str, any]]) -> Dict[str, any]:
        raise NotImplementedError()

    def __call__(self, score_detail: List[Dict[str, any]]) -> Dict[str, any]:
        if len(score_detail) > 0:
            for col in self.need_score_col:
                assert col in score_detail[0], ValueError(f"not found {col} score")
        try:
            return self._agg(score_detail)
        except Exception as e:
            return {"error": str(e)}


class WER(AggPolicy):
    def __init__(self, need_score_col: List[str] = None, ignore_case: bool = False):
        super().__init__(need_score_col)
        self.ignore_case = ignore_case

    def _agg(self, score_detail: List[Dict[str, any]]) -> Dict[str, float]:
        predl, refl = [str(item["pred"]) for item in score_detail], [
            str(item["ref"]) for item in score_detail
        ]
        if self.ignore_case:
            predl, refl = [item.lower() for item in predl], [
                item.lower() for item in refl
            ]
        return {"wer(%)": wer(refl, predl) * 100}


class PracticeWER(AggPolicy):
    def __init__(self, need_score_col: List[str] = None, lang: str = "en"):
        super().__init__(need_score_col)
        self.lang = lang
        if lang == "ja":
            self.lang = "ja-mecab"

    def _agg(self, score_detail: List[Dict[str, any]]) -> Dict[str, float]:
        predl, refl = [str(item["pred"]) for item in score_detail], [
            str(item["ref"]) for item in score_detail
        ]
        predl, refl = [item.lower() for item in predl], [item.lower() for item in refl]
        return {"wer(%)": compute_wer(refl, predl, self.lang) * 100}


class ACC(AggPolicy):

    def _agg(self, score_detail: List[Dict[str, any]]) -> Dict[str, float]:
        predl, refl = [str(item["pred"]) for item in score_detail], [
            str(item["ref"]) for item in score_detail
        ]
        return {"acc(%)": accuracy_score(refl, predl) * 100}


class NaiveACC(AggPolicy):
    def __init__(self, need_score_col: List[str] = None):
        super().__init__(need_score_col)

    def _agg(self, score_detail: List[Dict[str, any]]) -> Dict[str, float]:
        res = {}
        for item in self.need_score_col:
            valid_l = [c[item] for c in score_detail if c.get(item) is not None]
            res[f"{item}(%)"] = sum(valid_l) / len(valid_l) * 100
        return res


class CER(AggPolicy):
    def __init__(self, need_score_col: List[str] = None, ignore_case: bool = False):
        super().__init__(need_score_col)
        self.ignore_case = ignore_case

    def _agg(self, score_detail: List[Dict[str, any]]) -> Dict[str, float]:
        predl, refl = [str(item["pred"]) for item in score_detail], [
            str(item["ref"]) for item in score_detail
        ]
        if self.ignore_case:
            predl, refl = [item.lower() for item in predl], [
                item.lower() for item in refl
            ]
        return {"cer": cer(predl, refl)}


class Dump(AggPolicy):

    def _agg(self, score_detail: List[Dict[str, any]]) -> Dict[str, float]:
        return {}


class BLEU(AggPolicy):
    def __init__(self, need_score_col: List[str] = None, lang: str = "13a"):
        super().__init__(need_score_col)
        self.lang = "13a"
        if lang == "zh":
            self.lang = "zh"
        elif lang == "ja":
            self.lang = "ja-mecab"

    def _agg(self, score_detail: List[Dict[str, any]]) -> Dict[str, float]:
        predl, refl = [str(item["pred"]) for item in score_detail], [
            str(item["ref"]) for item in score_detail
        ]

        pred, ref = [], []
        for p, r in zip(predl, refl):
            if r:
                pred.append(p)
                ref.append(r)
        res = sacrebleu.corpus_bleu(pred, [ref], tokenize=self.lang)
        return {"bleu": res.score}


class Coco(AggPolicy):
    def _agg(self, score_detail: List[Dict[str, any]]) -> Dict[str, float]:
        predl, refl = [str(item["pred"]) for item in score_detail], [
            item["ref"] for item in score_detail
        ]

        pred, ref = [], []
        for p, r in zip(predl, refl):
            if r:
                pred.append(p)
                if isinstance(r, str):
                    r = [r]
                ref.append(r)
        res = compute_caption(ref, pred)
        return res
