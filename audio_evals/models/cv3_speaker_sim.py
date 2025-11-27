import logging
import os
import sys
from typing import Dict
from audio_evals.base import PromptStruct
from audio_evals.models.model import OfflineModel
from audio_evals.isolate import isolated
import select
import uuid

logger = logging.getLogger(__name__)


@isolated("audio_evals/lib/cv3_speaker_sim/speaker_sim.py")
class CV3SpeakerSim(OfflineModel):
    """
    Speaker similarity model using 3D-Speaker models.

    This model computes cosine similarity between speaker embeddings extracted
    from two audio files using models from the 3D-Speaker project.

    Args:
        model_id: Model identifier from the 3D-Speaker project.
                 Default: 'damo/speech_eres2net_base_sv_zh-cn_3dspeaker_16k'
                 Supported models:
                 - damo/speech_campplus_sv_en_voxceleb_16k
                 - damo/speech_campplus_sv_zh-cn_16k-common
                 - damo/speech_eres2net_sv_en_voxceleb_16k
                 - damo/speech_eres2net_sv_zh-cn_16k-common
                 - damo/speech_eres2net_base_sv_zh-cn_3dspeaker_16k
                 - damo/speech_eres2net_large_sv_zh-cn_3dspeaker_16k
        local_model_dir: Local directory where pretrained models are stored.
                        Default: 'pretrained'
        device: Device to run the model on. Default: 'cuda:0'
        sample_params: Additional sampling parameters (optional)

    Usage:
        model = CV3SpeakerSim(
            model_id='damo/speech_eres2net_base_sv_zh-cn_3dspeaker_16k',
            local_model_dir='/path/to/models'
        )

        prompt = {
            'audios': ['/path/to/audio1.wav', '/path/to/audio2.wav']
        }

        similarity = model._inference(prompt)
        # Returns: float in range [-1.0, 1.0]
    """

    def __init__(
        self,
        path: str,
        model_id: str = "damo/speech_eres2net_base_sv_zh-cn_3dspeaker_16k",
        sample_params: Dict = None,
    ):
        self.command_args = {
            "model_id": model_id,
            "path": path,
        }
        super().__init__(is_chat=False, sample_params=sample_params)

    def _inference(self, prompt: PromptStruct, **kwargs) -> float:
        """
        Compute speaker similarity between two audio files.

        Args:
            prompt: PromptStruct containing 'audios' key with list of 2 audio file paths
            **kwargs: Additional keyword arguments (unused)

        Returns:
            float: Cosine similarity score between -1.0 and 1.0

        Raises:
            AssertionError: If number of audio files is not exactly 2
            RuntimeError: If inference fails
        """
        audio_paths = prompt["audios"]
        assert (
            len(audio_paths) == 2
        ), f"CV3SpeakerSim requires exactly 2 audio files, got {len(audio_paths)}"

        # Generate unique identifier for this request
        uid = str(uuid.uuid4())
        prefix = f"{uid}->"

        # Wait for stdin to be ready and send request
        while True:
            _, wlist, _ = select.select([], [self.process.stdin], [], 60)
            if wlist:
                request = f"{prefix}{','.join(audio_paths)}\n"
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
                            # Extract similarity score
                            similarity_str = result[len(prefix) :]
                            similarity = float(similarity_str)

                            # Send close signal
                            self.process.stdin.write(f"{prefix}close\n")
                            self.process.stdin.flush()

                            logger.debug(f"Received similarity: {similarity}")
                            return similarity

                        elif result.startswith("Error:"):
                            error_msg = result[6:].strip()
                            raise RuntimeError(
                                f"Speaker similarity inference failed: {error_msg}"
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
                logger.error(f"Failed to parse similarity score: {str(e)}")
                raise RuntimeError(
                    f"Invalid response format from speaker similarity model"
                )
