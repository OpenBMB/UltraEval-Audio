import atexit
import logging
import os
import subprocess
import sys
import tempfile
import time
from typing import Dict
from audio_evals.base import PromptStruct
from audio_evals.models.model import OfflineModel
from audio_evals.isolate import isolated
import select


logger = logging.getLogger(__name__)


@isolated("audio_evals/lib/simo/simo.py")
class WavLM(OfflineModel):
    def __init__(self, path: str, sample_params: Dict = None):
        self.command_args = {
            "path": path,
        }
        super().__init__(is_chat=False, sample_params=sample_params)

    def _inference(self, prompt: PromptStruct, **kwargs) -> float:
        audio_paths = prompt["audios"]
        assert len(audio_paths) == 2, "wav lm must be used with two audio files."
        import uuid

        uid = str(uuid.uuid4())
        prefix = f"{uid}->"

        while True:
            _, wlist, _ = select.select([], [self.process.stdin], [], 60)
            if wlist:
                self.process.stdin.write(f"{prefix}{','.join(audio_paths)}\n")
                self.process.stdin.flush()
                print("already write in")
                break

        while True:
            rlist, _, _ = select.select(
                [self.process.stdout, self.process.stderr], [], [], 1
            )
            try:
                for stream in rlist:
                    if stream == self.process.stdout:
                        result = self.process.stdout.readline().strip()
                        if not result:
                            continue
                        if result.startswith(prefix):
                            self.process.stdin.write("{}close\n".format(prefix))
                            self.process.stdin.flush()
                            return float(result[len(prefix) :])
                        elif result.startswith("Error:"):
                            raise RuntimeError("wav lm failed: {}".format(result))
                        else:
                            logger.info(result)
                    elif stream == self.process.stderr:
                        err = self.process.stderr.readline().strip()
                        if err:
                            logger.error(f"Process stderr: {err}")
                # logger.info("Waiting for wav lm result...")
            except BlockingIOError as e:
                logger.error(f"BlockingIOError occurred: {str(e)}")
                continue
