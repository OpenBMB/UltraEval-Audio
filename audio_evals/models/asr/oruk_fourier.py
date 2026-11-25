from __future__ import annotations

import json
import logging
import select
import time
import uuid

from audio_evals.base import PromptStruct
from audio_evals.isolate import isolated
from audio_evals.models.model import OfflineModel

logger = logging.getLogger(__name__)


def build_request(
    prompt: PromptStruct,
    default_language: str | None = None,
    **kwargs,
) -> dict[str, str]:
    if not isinstance(prompt, dict) or not prompt.get("audio"):
        raise ValueError("OrukFourierASR requires an audio path in prompt['audio']")

    request: dict[str, str] = {"audio": prompt["audio"]}
    language = prompt.get("language") or kwargs.get("language") or default_language
    if language:
        request["language"] = language
    return request


@isolated("audio_evals/lib/OrukFourier/main.py")
class OrukFourierASR(OfflineModel):
    def __init__(
        self,
        path: str = "oruk/oruk-fourier-2b",
        model_revision: str = "8c2a8871f1304aa2f634da4fb95e24438c4945b9",
        dtype: str = "bfloat16",
        device_map: str = "cuda:0",
        max_new_tokens: int = 512,
        language: str | None = "en",
        sample_params: dict | None = None,
        *args,
        **kwargs,
    ):
        self.default_language = language
        self.command_args = {
            "path": path,
            "model_revision": model_revision,
            "dtype": dtype,
            "device_map": device_map,
            "max_new_tokens": max_new_tokens,
        }
        if language:
            self.command_args["language"] = language
        super().__init__(is_chat=False, sample_params=sample_params)

    def _inference(self, prompt: PromptStruct, **kwargs) -> str:
        request = build_request(
            prompt, default_language=self.default_language, **kwargs
        )
        prefix = f"{uuid.uuid4()}->"
        message = json.dumps(request, ensure_ascii=False)

        _, writable, _ = select.select([], [self.process.stdin], [], 60)
        if not writable:
            raise TimeoutError("Timed out writing a request to OrukFourierASR")
        self.process.stdin.write(f"{prefix}{message}\n")
        self.process.stdin.flush()

        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(
                    f"OrukFourierASR worker exited with code {self.process.returncode}"
                )

            readable, _, _ = select.select(
                [self.process.stdout, self.process.stderr], [], [], 1.0
            )
            for stream in readable:
                line = stream.readline().strip()
                if not line:
                    continue
                if stream is self.process.stderr:
                    logger.error("OrukFourierASR worker: %s", line)
                    continue
                if line.startswith(prefix):
                    self.process.stdin.write(f"{prefix}close\n")
                    self.process.stdin.flush()
                    return line[len(prefix) :]
                if line.startswith("Error:"):
                    raise RuntimeError(f"OrukFourierASR failed: {line[6:].strip()}")
                logger.info(line)

        raise TimeoutError("Timed out waiting for OrukFourierASR inference")
