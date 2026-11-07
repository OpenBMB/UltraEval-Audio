"""
dots.tts model wrapper for UltraEval-Audio.

Covers the dots.tts checkpoint family (base / soar / mf*) — all share the
official PyPI package `dots.tts` runtime API (DotsTtsRuntime.generate), only
the checkpoint `path` and sampling params differ.

Reference: https://huggingface.co/dots-studio/dots.tts-soar
"""

import json
import logging
import os
import select
from typing import Dict, Optional

from audio_evals.base import PromptStruct
from audio_evals.isolate import isolated
from audio_evals.models.model import OfflineModel

logger = logging.getLogger(__name__)


@isolated("audio_evals/lib/DotsTTS/main.py")
class DotsTTS(OfflineModel):
    def __init__(
        self,
        path: str,
        dtype: str = "bfloat16",
        device: str = "cuda:0",
        optimize: bool = False,
        sample_params: Optional[Dict] = None,
        *args,
        **kwargs,
    ):
        if not os.path.exists(path):
            path = self._download_model(path)

        self.command_args = {
            "path": path,
            "dtype": dtype,
            "device": device,
        }
        if optimize:
            self.command_args["optimize"] = ""
        super().__init__(is_chat=True, sample_params=sample_params)

    def _inference(self, prompt: PromptStruct, **kwargs) -> str:
        import uuid

        uid = str(uuid.uuid4())
        prefix = f"{uid}->"

        if isinstance(prompt, dict):
            prompt.update(kwargs)
        else:
            prompt = {"text": prompt, **kwargs}

        while True:
            _, wlist, _ = select.select([], [self.process.stdin], [], 180)
            if not wlist:
                raise RuntimeError("Write timeout after 180 seconds")
            try:
                self.process.stdin.write(
                    f"{prefix}{json.dumps(prompt, ensure_ascii=False)}\n"
                )
                self.process.stdin.flush()
                logger.debug("prompt written to dots.tts stdin")
                break
            except BlockingIOError:
                continue

        # No ready-wait at init or per request: cold start (venv install +
        # dots.tts import + model load) is covered by a long per-request
        # timeout, and stdin buffers prompts until the child is up.
        while True:
            rlist, _, _ = select.select(
                [self.process.stdout, self.process.stderr], [], [], 1800
            )
            if not rlist:
                err_msg = "Read timeout after 1800 seconds"
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
                            raise RuntimeError(f"dots.tts failed: {result}")
                        else:
                            logger.info(result)
                    elif stream == self.process.stderr:
                        err = self.process.stderr.readline().strip()
                        if err:
                            logger.error(f"Process stderr: {err}")
            except BlockingIOError as e:
                logger.error(f"BlockingIOError occurred: {str(e)}")
                continue
