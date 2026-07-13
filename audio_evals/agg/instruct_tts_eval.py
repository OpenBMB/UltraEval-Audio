from typing import Dict, List

from audio_evals.agg.base import AggPolicy

_INSTRUCTION_TYPES = ("APS", "DSD", "RP")


class InstructTTSEvalAgg(AggPolicy):
    """
    Aggregation policy for InstructTTSEval.

    Groups per-sample scores by ``instruction_type`` (APS / DSD / RP) and
    computes the percentage of samples where Gemini judged the generated audio
    as consistent with the style instruction (``gemini_score == 1``).

    Null scores (failed Gemini calls) are excluded from the denominator so
    they do not penalise the final rate; instead ``null_count`` is reported.

    Output keys
    -----------
    APS(%)   : pass-rate for Acoustic-Parameter Specification
    DSD(%)   : pass-rate for Descriptive-Style Directive
    RP(%)    : pass-rate for Role-Play
    AVG(%)   : macro average of the three pass-rates
    """

    def _agg(self, score_detail: List[Dict[str, any]]) -> Dict[str, any]:
        buckets: Dict[str, List[int]] = {t: [] for t in _INSTRUCTION_TYPES}
        null_counts: Dict[str, int] = {t: 0 for t in _INSTRUCTION_TYPES}

        for item in score_detail:
            inst_type = item.get("instruction_type", "")
            score = item.get("gemini_score")

            if inst_type not in buckets:
                continue

            if score is None:
                null_counts[inst_type] += 1
            else:
                buckets[inst_type].append(int(score))

        result: Dict[str, any] = {}
        valid_rates = []

        for inst_type in _INSTRUCTION_TYPES:
            scores = buckets[inst_type]
            nulls = null_counts[inst_type]
            total = len(scores) + nulls

            if total == 0:
                continue

            if scores:
                rate = sum(scores) / len(scores) * 100
            else:
                rate = 0.0

            result[f"{inst_type}(%)"] = round(rate, 2)
            result[f"{inst_type}_valid"] = len(scores)
            result[f"{inst_type}_null"] = nulls
            valid_rates.append(rate)

        if valid_rates:
            result["AVG(%)"] = round(sum(valid_rates) / len(valid_rates), 2)

        return result
