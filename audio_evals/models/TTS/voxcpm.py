import json
import logging
import select
from typing import Dict

from audio_evals.base import PromptStruct
from audio_evals.models.model import OfflineModel
from audio_evals.isolate import isolated
import os

logger = logging.getLogger(__name__)


@isolated("audio_evals/lib/VoxCPM/main.py")
class VoxCPM(OfflineModel):

    def __init__(
        self,
        path: str,
        vc_mode: bool,
        denoise: bool,
        sample_params: Dict = None,
        *args,
        **kwargs,
    ):

        if not os.path.exists(path):
            path = self._download_model(path)

        self.command_args = {"path": path}
        if vc_mode:
            self.command_args["vc_mode"] = ""
        if denoise:
            self.command_args["denoise"] = ""
        super().__init__(is_chat=True, sample_params=sample_params)

    def _inference(self, prompt: PromptStruct, **kwargs):
        import uuid

        uid = str(uuid.uuid4())
        prefix = f"{uid}->"
        prompt.update(kwargs)

        while True:
            _, wlist, _ = select.select([], [self.process.stdin], [], 180)
            if wlist:
                self.process.stdin.write(f"{prefix}{json.dumps(prompt)}\n")
                self.process.stdin.flush()
                logger.debug("prompt written to VoxCPM stdin")
                break

        while True:
            rlist, _, _ = select.select(
                [self.process.stdout, self.process.stderr], [], [], 180
            )
            if not rlist:
                err_msg = "Read timeout after 60 seconds"
                logger.error(err_msg)
                raise RuntimeError(err_msg)

            try:
                for stream in rlist:
                    if stream == self.process.stdout:
                        result = self.process.stdout.readline().strip()
                        if not result:
                            continue
                        if result.startswith(prefix):
                            self.process.stdin.write(f"{prefix}close\n")
                            self.process.stdin.flush()
                            return result[len(prefix) :]
                        elif result.startswith("Error:"):
                            raise RuntimeError(f"VoxCPM failed: {result}")
                        else:
                            logger.info(result)
                    elif stream == self.process.stderr:
                        err = self.process.stderr.readline().strip()
                        if err:
                            logger.error(f"Process stderr: {err}")
            except BlockingIOError as e:
                logger.error(f"BlockingIOError occurred: {str(e)}")
