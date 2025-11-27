from typing import Dict, List
from audio_evals.agg.base import AggPolicy
from audio_evals.lib.wer import compute_wer
import numpy as np


class PracticeWERFilter(AggPolicy):
    def __init__(self, need_score_col: List[str] = None, lang: str = "13a"):
        super().__init__(need_score_col)
        self.lang = lang

    def _agg(self, score_detail: List[Dict[str, any]]) -> Dict[str, float]:
        label_wer = "cer%" if "cer%" in score_detail[0] else "wer%"
        score_detail = [item for item in score_detail if item[label_wer] < 100]
        predl, refl = [str(item["pred"]) for item in score_detail], [
            str(item["ref"]) for item in score_detail
        ]
        predl, refl = [item.lower() for item in predl], [item.lower() for item in refl]
        return {"wer(%)": compute_wer(refl, predl, self.lang) * 100}


class WeightedAverage(AggPolicy):
    def __init__(self, weight_col: str, need_score_col: List[str] = None):
        super().__init__(need_score_col)
        self.weight_col = weight_col

    def _agg(self, score_detail: List[Dict[str, any]]) -> Dict[str, float]:
        res = {}
        if not self.need_score_col:
            for k, v in score_detail[0].items():
                if isinstance(v, (int, float)) and k != self.weight_col:
                    self.need_score_col.append(k)
                else:
                    print(f"ignore {k} as it is not a number, but {v}")
        for item in self.need_score_col:
            valid_l = [c[item] for c in score_detail if c.get(item) is not None]
            weight_l = [c[self.weight_col] for c in score_detail]
            if "%" in item:
                res[item] = float(
                    sum(np.array(valid_l) * np.array(weight_l)) / sum(weight_l)
                )
            else:
                res[f"{item}(%)"] = float(
                    sum(np.array(valid_l) * np.array(weight_l)) / sum(weight_l) * 100
                )
        return res
