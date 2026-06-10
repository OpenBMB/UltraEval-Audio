import json
import logging
import os
import select
from typing import Dict

from audio_evals.base import PromptStruct
from audio_evals.isolate import isolated
from audio_evals.models.model import OfflineModel


logger = logging.getLogger(__name__)


@isolated("audio_evals/lib/Qwen3ASR/main.py")
class Qwen3ASR(OfflineModel):
    def __init__(self, path: str, sample_params: Dict = None, *args, **kwargs):
        if not os.path.exists(path):
            path = self._download_model(path)

        self.command_args = {
            "path": path,
        }
        for key in [
            "dtype",
            "device_map",
            "max_new_tokens",
            "max_inference_batch_size",
        ]:
            if kwargs.get(key) is not None:
                self.command_args[key] = kwargs[key]

        super().__init__(is_chat=False, sample_params=sample_params)

    def _inference(self, prompt: PromptStruct, **kwargs) -> str:
        payload = {"audio": prompt["audio"]}
        # language is a per-request transcribe param, sourced from the prompt
        # (falling back to sample_params); empty/None means auto-detect.
        language = prompt.get("language") or kwargs.get("language")
        if language:
            payload["language"] = language

        import uuid

        uid = str(uuid.uuid4())
        prefix = f"{uid}->"
        message = json.dumps(payload, ensure_ascii=False)

        while True:
            _, wlist, _ = select.select([], [self.process.stdin], [], 60)
            if wlist:
                self.process.stdin.write(f"{prefix}{message}\n")
                self.process.stdin.flush()
                break

        while True:
            reads, _, _ = select.select(
                [self.process.stdout, self.process.stderr], [], [], 1.0
            )
            for read in reads:
                if read is self.process.stdout:
                    result = self.process.stdout.readline().strip()
                    if result:
                        if result.startswith(prefix):
                            self.process.stdin.write("{}close\n".format(prefix))
                            self.process.stdin.flush()
                            return result[len(prefix) :]
                        elif result.startswith("Error:"):
                            raise RuntimeError("Qwen3-ASR failed: {}".format(result))
                        else:
                            logger.info(result)
                if read is self.process.stderr:
                    error_output = self.process.stderr.readline()
                    if error_output:
                        print(f"stderr: {error_output.strip()}")
