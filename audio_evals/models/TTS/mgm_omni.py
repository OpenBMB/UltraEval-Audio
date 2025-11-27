import json
import logging
import select
from typing import Dict

from audio_evals.base import PromptStruct
from audio_evals.models.model import OfflineModel
from audio_evals.isolate import isolated
import os

logger = logging.getLogger(__name__)


@isolated(
    "audio_evals/lib/MGM_Omni/main.py",
    pre_command="mkdir -p ./third_party && ([ ! -d './third_party/MGM-Omni' ] && git clone https://github.com/dvlab-research/MGM-Omni.git ./third_party/MGM-Omni &&"
    "cd ./third_party/MGM-Omni && git submodule update --init --recursive && cd ../..) || true",
)
class MGMOmni(OfflineModel):

    def __init__(
        self,
        path: str,
        lang: str = "en",
        load_8bit: bool = False,
        load_4bit: bool = False,
        sample_params: Dict = None,
        *args,
        **kwargs,
    ):

        if not os.path.exists(path):
            path = self._download_model(path)

        self.command_args = {
            "path": path,
            "lang": lang,
        }

        if load_8bit:
            self.command_args["load_8bit"] = ""

        if load_4bit:
            self.command_args["load_4bit"] = ""

        super().__init__(is_chat=True, sample_params=sample_params)

    def _inference(self, prompt: PromptStruct, **kwargs):
        import uuid

        uid = str(uuid.uuid4())
        prefix = f"{uid}->"

        while True:
            _, wlist, _ = select.select([], [self.process.stdin], [], 60)
            if wlist:
                self.process.stdin.write(f"{prefix}{json.dumps(prompt)}\n")
                self.process.stdin.flush()
                logger.debug("prompt written to MGM-Omni stdin")
                break

        while True:
            rlist, _, _ = select.select(
                [self.process.stdout, self.process.stderr], [], [], 300
            )
            if not rlist:
                err_msg = "Read timeout after 300 seconds"
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
                            raise RuntimeError(f"MGM-Omni failed: {result}")
                        else:
                            logger.info(result)
                    elif stream == self.process.stderr:
                        err = self.process.stderr.readline().strip()
                        if err:
                            logger.error(f"Process stderr: {err}")
            except BlockingIOError as e:
                logger.error(f"BlockingIOError occurred: {str(e)}")
