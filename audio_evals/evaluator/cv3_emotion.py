import os
from typing import Dict

from audio_evals.evaluator.base import Evaluator


class CV3EmotionEval(Evaluator):
    """
    Emotion evaluation for CV3-Eval benchmark.

    This evaluator uses emotion2vec_plus_large to recognize emotions from
    generated speech and compares with the reference emotion label.

    The reference emotion is expected to be provided in the kwargs (e.g., from the
    utterance ID format: {emotion}_{level}_... in CV3-Eval dataset).

    Args:
        model_name: Name of the emotion recognition model.
                   Default: 'emotion2vec_plus_large'
    """

    def __init__(
        self,
        model_name: str = "emotion2vec_plus_large",
    ):
        from audio_evals.registry import registry

        self.model = registry.get_model(model_name)

    def _eval(self, pred, label, **kwargs) -> Dict[str, any]:
        """
        Evaluate emotion accuracy.

        Args:
            pred: Path to the generated audio file
            label: Reference label (may contain emotion info or use ref_emotion from kwargs)
            **kwargs: Additional arguments, may contain:
                - ref_emotion: Reference emotion label
                - utterance_id: Utterance ID in format {emotion}_{level}_...

        Returns:
            Dict containing:
                - emotion_pred: Predicted emotion
                - emotion_ref: Reference emotion
                - emotion_score: Confidence score from model
                - emotion_match: 1 if match, 0 otherwise
        """
        pred = str(pred)
        assert os.path.exists(pred), f"Prediction file {pred} does not exist"

        # Get reference emotion from kwargs
        ref_emotion = kwargs.get("ref_emotion", None)

        # If ref_emotion not provided, try to extract from utterance_id or label
        if ref_emotion is None:
            utterance_id = kwargs.get("utterance_id", kwargs.get("id", ""))
            if utterance_id and "_" in str(utterance_id):
                # CV3-Eval format: {emotion}_{level}_...
                ref_emotion = str(utterance_id).split("_")[0].lower()

        # Run emotion recognition on generated audio
        emotion_pred, emotion_score = self.model.inference({"audio": pred})
        emotion_pred = emotion_pred.lower()

        # Normalize emotion labels for comparison
        ref_emotion_normalized = ref_emotion.lower() if ref_emotion else "unknown"

        # Check if prediction matches reference
        emotion_match = 1 if emotion_pred == ref_emotion_normalized else 0

        return {
            "emotion_pred": emotion_pred,
            "emotion_ref": ref_emotion_normalized,
            "emotion_score": emotion_score,
            "emotion_match": emotion_match,
            "pred": pred,
            "ref": label,
        }
