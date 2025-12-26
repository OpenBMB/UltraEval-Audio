import logging
import os
import select
import uuid
from typing import Dict

from audio_evals.base import PromptStruct
from audio_evals.models.model import OfflineModel
from audio_evals.isolate import isolated


logger = logging.getLogger(__name__)


@isolated("audio_evals/lib/ChatTTS/encodec.py")
class ChatTTSModel(OfflineModel):
    def __init__(
        self,
        model_path: str,
        sample_params: Dict[str, any] = None,
        *args,
        **kwargs,
    ):
        if not os.path.exists(model_path):
            # If there's a download logic, it could go here
            pass
        self.command_args = {
            "model_path": model_path,
        }
        super().__init__(is_chat=False, sample_params=sample_params)

    def _inference(self, prompt: PromptStruct, **kwargs) -> str:
        audio_path = prompt["audio"]
        uid = str(uuid.uuid4())
        prefix = f"{uid}->"

        # Wait for stdin to be writable
        _, wlist, _ = select.select([], [self.process.stdin], [], 60)
        if not wlist:
            raise RuntimeError("ChatTTS: timeout waiting for stdin to be writable")

        self.process.stdin.write(f"{prefix}{audio_path}\n")
        self.process.stdin.flush()

        while True:
            reads, _, _ = select.select(
                [self.process.stdout, self.process.stderr], [], [], 60
            )
            if not reads:
                raise RuntimeError("ChatTTS: timeout waiting for response")

            for read in reads:
                if read is self.process.stdout:
                    result = self.process.stdout.readline().strip()
                    if result.startswith(prefix):
                        # Send close signal to acknowledge receipt
                        self.process.stdin.write(f"{prefix}close\n")
                        self.process.stdin.flush()
                        return result[len(prefix) :]
                    elif result.startswith("Error:"):
                        raise RuntimeError("ChatTTS encoding failed: {}".format(result))
                    elif result:
                        logger.info(result)
                if read is self.process.stderr:
                    error_output = self.process.stderr.readline()
                    if error_output:
                        logger.warning(f"stderr: {error_output.strip()}")
