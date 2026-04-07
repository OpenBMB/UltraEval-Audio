import logging
import os
import sys
from typing import Dict, Tuple
from audio_evals.base import PromptStruct
from audio_evals.models.model import OfflineModel
from audio_evals.isolate import isolated
import select
import uuid

logger = logging.getLogger(__name__)


@isolated("audio_evals/lib/emotion2vec/main.py")
class Emotion2Vec(OfflineModel):
    """
    Emotion recognition model using emotion2vec_plus_large from ModelScope.

    This model recognizes emotions from audio files using the emotion2vec model.

    Supported emotion labels:
        - angry, disgusted, fearful, happy, neutral, other, sad, surprised, unk

    Args:
        model: Model identifier for emotion2vec. Default: 'iic/emotion2vec_plus_large'
        sample_params: Additional sampling parameters (optional)

    Usage:
        model = Emotion2Vec(model='iic/emotion2vec_plus_large')

        prompt = {
            'audio': '/path/to/audio.wav'
        }

        result = model._inference(prompt)
        # Returns: Tuple[str, float] - (predicted_emotion, confidence_score)
    """

    def __init__(
        self,
        model: str = "iic/emotion2vec_plus_large",
        sample_params: Dict = None,
    ):
        if not os.path.exists(model):
            model = self._download_model_from_modelscope(model)

        self.command_args = {
            "model": model,
        }
        super().__init__(is_chat=False, sample_params=sample_params)

    def _inference(self, prompt: PromptStruct, **kwargs) -> Tuple[str, float]:
        """
        Predict emotion from an audio file.

        Args:
            prompt: PromptStruct containing 'audio' key with audio file path
            **kwargs: Additional keyword arguments (unused)

        Returns:
            Tuple[str, float]: (predicted_emotion, confidence_score)

        Raises:
            RuntimeError: If inference fails
        """
        audio_path = prompt["audio"]

        # Generate unique identifier for this request
        uid = str(uuid.uuid4())
        prefix = f"{uid}->"

        # Wait for stdin to be ready and send request
        while True:
            _, wlist, _ = select.select([], [self.process.stdin], [], 60)
            if wlist:
                request = f"{prefix}{audio_path}\n"
                self.process.stdin.write(request)
                self.process.stdin.flush()
                logger.debug(f"Sent request: {request.strip()}")
                break

        # Read response
        while True:
            rlist, _, _ = select.select(
                [self.process.stdout, self.process.stderr], [], [], 60
            )

            try:
                for stream in rlist:
                    if stream == self.process.stdout:
                        result = self.process.stdout.readline().strip()
                        if not result:
                            continue

                        if result.startswith(prefix):
                            # Extract emotion and score: format is "emotion,score"
                            result_str = result[len(prefix) :]
                            parts = result_str.split(",")
                            emotion = parts[0]
                            score = float(parts[1]) if len(parts) > 1 else 0.0

                            # Send close signal
                            self.process.stdin.write(f"{prefix}close\n")
                            self.process.stdin.flush()

                            logger.debug(f"Received emotion: {emotion}, score: {score}")
                            return (emotion, score)

                        elif result.startswith("Error:"):
                            error_msg = result[6:].strip()
                            raise RuntimeError(
                                f"Emotion recognition failed: {error_msg}"
                            )

                        else:
                            # Log intermediate messages
                            logger.info(result)

                    elif stream == self.process.stderr:
                        err = self.process.stderr.readline().strip()
                        if err:
                            logger.error(f"Process stderr: {err}")

            except BlockingIOError as e:
                logger.error(f"BlockingIOError occurred: {str(e)}")
                continue
            except ValueError as e:
                logger.error(f"Failed to parse emotion result: {str(e)}")
                raise RuntimeError(
                    f"Invalid response format from emotion recognition model"
                )
