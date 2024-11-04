from typing import Dict, List
from audio_evals.agg.base import AggPolicy


class AirChat(AggPolicy):

    def _agg(self, score_detail: List[Dict[str, any]]) -> Dict[str, float]:
        predl, refl = [item["pred_score"] for item in score_detail], [
            item["ref_score"] for item in score_detail
        ]
        win_count = sum([1 for i in range(len(predl)) if predl[i] > refl[i]])
        return {
            "win(%)": win_count / len(predl) * 100,
            "ref_score": sum(refl) / len(refl),
            "pred_score": sum(predl) / len(predl),
        }
