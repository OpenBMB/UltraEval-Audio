"""FireRedTTS3 Base and Instruct wrappers."""

import json
import logging
import os
import select
import uuid
from typing import Dict, Optional

from audio_evals.base import PromptStruct
from audio_evals.device import resolve_device
from audio_evals.isolate import isolated
from audio_evals.models.model import OfflineModel


logger = logging.getLogger(__name__)

_UPSTREAM_COMMIT = "b2ec09a41c3ac89dad8d209391664057a4a1f94b"
_PRE_COMMAND = f"""mkdir -p ./third_party && ([ ! -d './third_party/FireRedTTS3' ] && \
GIT_LFS_SKIP_SMUDGE=1 git init ./third_party/FireRedTTS3 && \
cd ./third_party/FireRedTTS3 && \
git remote add origin https://github.com/FireRedTeam/FireRedTTS3.git && \
git fetch --depth 1 origin {_UPSTREAM_COMMIT} && \
git checkout FETCH_HEAD && cd ../..) || true"""


@isolated(
    "audio_evals/lib/FireRedTTS3/main.py",
    pre_command=_PRE_COMMAND,
)
class FireRedTTS3(OfflineModel):
    def __init__(
        self,
        path: str,
        mode: str = "base",
        device: str = "cuda:0",
        use_wetext: bool = True,
        sample_params: Optional[Dict] = None,
        *args,
        **kwargs,
    ):
        if mode not in ("base", "instruct"):
            raise ValueError(f"Unsupported FireRedTTS3 mode: {mode}")
        if not os.path.exists(path):
            path = self._download_model(path)
        self.command_args = {
            "path": path,
            "source-path": "./third_party/FireRedTTS3",
            "mode": mode,
            "device": resolve_device(device),
        }
        if use_wetext:
            self.command_args["use-wetext"] = ""
        super().__init__(is_chat=True, sample_params=sample_params)

    def _inference(self, prompt: PromptStruct, **kwargs) -> str:
        uid = str(uuid.uuid4())
        prefix = f"{uid}->"
        if isinstance(prompt, dict):
            payload = {**prompt, **kwargs}
        else:
            payload = {"text": prompt, **kwargs}

        while True:
            _, writable, _ = select.select([], [self.process.stdin], [], 180)
            if not writable:
                raise RuntimeError("Write timeout after 180 seconds")
            try:
                self.process.stdin.write(
                    f"{prefix}{json.dumps(payload, ensure_ascii=False)}\n"
                )
                self.process.stdin.flush()
                break
            except BlockingIOError:
                continue

        while True:
            readable, _, _ = select.select(
                [self.process.stdout, self.process.stderr], [], [], 1800
            )
            if not readable:
                raise RuntimeError("Read timeout after 1800 seconds")
            for stream in readable:
                if stream == self.process.stdout:
                    result = self.process.stdout.readline().strip()
                    if result.startswith(prefix):
                        self.process.stdin.write(f"{prefix}close\n")
                        self.process.stdin.flush()
                        return result[len(prefix) :]
                    if result.startswith("Error:"):
                        raise RuntimeError(f"FireRedTTS3 failed: {result}")
                else:
                    message = self.process.stderr.readline().strip()
                    if message:
                        logger.info("FireRedTTS3: %s", message)
