import json
import logging
import select
import time
import uuid
from typing import Dict

from audio_evals.base import PromptStruct
from audio_evals.device import resolve_device
from audio_evals.isolate import isolated
from audio_evals.models.model import OfflineModel


logger = logging.getLogger(__name__)


def build_request(prompt: PromptStruct, **kwargs) -> Dict[str, str]:
    if not isinstance(prompt, dict) or not prompt.get("audio"):
        raise ValueError("Fun-ASR-Nano requires an audio path in prompt['audio']")

    request = {"audio": prompt["audio"]}
    language = prompt.get("language") or kwargs.get("language")
    if language:
        request["language"] = language
    return request


@isolated("audio_evals/lib/FunASRNano/main.py")
class FunASRNano(OfflineModel):
    def __init__(
        self,
        path: str,
        hub: str = "hf",
        model_revision: str = "main",
        device: str = "cuda:0",
        sample_params: Dict = None,
        *args,
        **kwargs,
    ):
        self.command_args = {
            "path": path,
            "hub": hub,
            "model_revision": model_revision,
            "device": resolve_device(device),
        }
        super().__init__(is_chat=False, sample_params=sample_params)

    def _inference(self, prompt: PromptStruct, **kwargs) -> str:
        request = build_request(prompt, **kwargs)
        prefix = f"{uuid.uuid4()}->"
        message = json.dumps(request, ensure_ascii=False)

        _, writable, _ = select.select([], [self.process.stdin], [], 60)
        if not writable:
            raise TimeoutError("Timed out writing a request to Fun-ASR-Nano")
        self.process.stdin.write(f"{prefix}{message}\n")
        self.process.stdin.flush()

        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(
                    f"Fun-ASR-Nano worker exited with code {self.process.returncode}"
                )

            readable, _, _ = select.select(
                [self.process.stdout, self.process.stderr], [], [], 1.0
            )
            for stream in readable:
                line = stream.readline().strip()
                if not line:
                    continue
                if stream is self.process.stderr:
                    logger.error("Fun-ASR-Nano worker: %s", line)
                    continue
                if line.startswith(prefix):
                    self.process.stdin.write(f"{prefix}close\n")
                    self.process.stdin.flush()
                    return line[len(prefix) :]
                if line.startswith("Error:"):
                    raise RuntimeError(f"Fun-ASR-Nano failed: {line[6:].strip()}")
                logger.info(line)

        raise TimeoutError("Timed out waiting for Fun-ASR-Nano inference")
