import json
import logging
import select
import uuid
from typing import Dict

from audio_evals.base import PromptStruct
from audio_evals.models.model import OfflineModel
from audio_evals.isolate import isolated

logger = logging.getLogger(__name__)


@isolated("audio_evals/lib/sherpa-onnx/main.py")
class SherpaOnnx(OfflineModel):
    def __init__(self, tokens: str, sample_params: Dict = None, *args, **kwargs):
        self.command_args = {
            "tokens": tokens,
        }
        for k, v in kwargs.items():
            if k == "offline":
                if v:
                    v = ""
                else:
                    continue

            self.command_args[k] = v
        super().__init__(is_chat=True, sample_params=sample_params)

    def _inference(self, prompt: PromptStruct, **kwargs) -> str:
        audio = prompt["audio"]
        uid = str(uuid.uuid4())
        prefix = f"{uid}->"

        # Send request
        while True:
            _, wlist, _ = select.select([], [self.process.stdin], [], 60)
            if wlist:
                request = json.dumps({"audio": audio})
                self.process.stdin.write(f"{prefix}{request}\n")
                self.process.stdin.flush()
                break

        # Receive response
        while True:
            reads, _, _ = select.select(
                [self.process.stdout, self.process.stderr], [], [], 1.0
            )
            for read in reads:
                if read is self.process.stdout:
                    result = self.process.stdout.readline().strip()
                    if result:
                        if result.startswith(prefix):
                            # Close the request
                            self.process.stdin.write(f"{prefix}close\n")
                            self.process.stdin.flush()

                            # Parse and return the result
                            response = json.loads(result[len(prefix) :])
                            return response["text"]
                        elif result.startswith("Error:"):
                            raise RuntimeError(f"SherpaOnnx failed: {result}")
                        else:
                            logger.info(result)
                if read is self.process.stderr:
                    error_output = self.process.stderr.readline()
                    if error_output:
                        logger.error(f"stderr: {error_output.strip()}")
