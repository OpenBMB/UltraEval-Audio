from typing import Dict, List

from audio_evals.agg.base import AggPolicy

# Official benchmark category order (used for stable output ordering).
_CATEGORY_ORDER = (
    "Emotions",
    "Paralinguistics",
    "Foreign Words",
    "Syntactic Complexity",
    "Pronunciation",
    "Questions",
)


def _mean(values: List[float]):
    return sum(values) / len(values) if values else None


class EmergentTTSEvalAgg(AggPolicy):
    """
    Aggregation policy for EmergentTTS-Eval.

    Computes the official win rate against the baseline.  A judge tie counts
    as 0.5, matching ``calculate_metrics_win_rate`` in the official repository.
    Samples whose judge call failed (``win is None``) are excluded from the
    denominator and reported through ``JudgeFailCount``.
    """

    def _agg(self, score_detail: List[Dict[str, any]]) -> Dict[str, any]:
        result: Dict[str, any] = {}

        wins: List[float] = []
        ties: List[float] = []
        judged = 0
        judge_fail = 0

        buckets: Dict[str, Dict[str, List[float]]] = {}

        for item in score_detail:
            category = item.get("category") or "Unknown"
            bucket = buckets.setdefault(
                category, {"win": [], "tie": [], "judge_fail": 0}
            )

            win = item.get("win")
            if win is None:
                judge_fail += 1
                bucket["judge_fail"] += 1
            else:
                judged += 1
                wins.append(float(win))
                bucket["win"].append(float(win))
                tie = item.get("tie")
                if tie is not None:
                    ties.append(float(tie))
                    bucket["tie"].append(float(tie))

        result["Total"] = len(score_detail)
        result["Judged"] = judged
        result["JudgeFailCount"] = judge_fail
        if judged:
            result["WinRate(%)"] = round(_mean(wins) * 100, 2)
            result["TieRate(%)"] = round(_mean(ties) * 100, 2) if ties else 0.0

        ordered_categories = [c for c in _CATEGORY_ORDER if c in buckets]
        ordered_categories.extend(
            sorted(c for c in buckets if c not in _CATEGORY_ORDER)
        )

        for category in ordered_categories:
            bucket = buckets[category]
            total = len(bucket["win"]) + bucket["judge_fail"]
            result[f"{category} Total"] = total
            result[f"{category} JudgeFailCount"] = bucket["judge_fail"]
            if bucket["win"]:
                result[f"{category} WinRate(%)"] = round(_mean(bucket["win"]) * 100, 2)
                result[f"{category} TieRate(%)"] = (
                    round(_mean(bucket["tie"]) * 100, 2) if bucket["tie"] else 0.0
                )

        return result
