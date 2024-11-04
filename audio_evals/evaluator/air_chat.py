from typing import Dict

from audio_evals.evaluator.base import Evaluator


prompt = (
    "You are a helpful and precise assistant for checking the quality of the answer.\n"
    "[Detailed Audio Description]\n{meta_info}\n[Question]\n{question}\n"
    "[The Start of Assistant 1s Answer]\n{label}\n[The End of Assistant 1s Answer]\n"
    "[The Start of Assistant 2s Answer]\n{pred}\n[The End of Assistant 2s Answer]\n[System]\n"
    "We would like to request your feedback on the performance of two AI assistants in response to the user question "
    "and audio description displayed above. AI assistants are provided with detailed audio descriptions and questions.\n"
    "Please rate the helpfulness, relevance, accuracy, and comprehensiveness of their responses. "
    "Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance. "
    "Please output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. "
    "The two scores are separated by a space."
)


class AIRChatEvaluator(Evaluator):
    def __init__(self, model_name: str):
        self.model_name = model_name

    def _eval(self, pred, label, **kwargs) -> Dict[str, any]:
        from audio_evals.registry import registry

        model = registry.get_model(self.model_name)
        p = prompt.format(
            meta_info=kwargs["meta_info"],
            question=kwargs["question"],
            label=label,
            pred=pred,
        )
        res = model.inference(p)
        ref_score, pred_score = res.split(" ")[0], res.split(" ")[1]
        return {
            "pred_score": float(pred_score),
            "ref_score": float(ref_score),
            "pred": pred,
            "ref": label,
        }
