import json
import logging
import os
import select
from typing import Dict

from audio_evals.base import PromptStruct
from audio_evals.models.model import OfflineModel
from audio_evals.constants import DEFAULT_MODEL_PATH

logger = logging.getLogger(__name__)


from audio_evals.isolate import isolated


@isolated("audio_evals/lib/whisper/main.py")
class WhisperModel(OfflineModel):
    def __init__(
        self,
        path: str = "openai/whisper-large-v3",
        sample_params: Dict[str, any] = None,
    ):
        if path.startswith("openai/") and not os.path.exists(path):
            path = self._download_model(path)

        self.command_args = {
            "path": path,
        }
        super().__init__(is_chat=True, sample_params=sample_params)

    def _process_prompt(self, prompt: PromptStruct) -> Dict[str, str]:
        if isinstance(prompt, list):
            for content in prompt:
                for line in content["contents"]:
                    if line["type"] == "audio":
                        if not os.path.exists(line["value"]):
                            raise FileNotFoundError(
                                f"Audio file not found: {line['value']}"
                            )
                        return {"audio": line["value"]}
        return prompt

    def _inference(self, prompt: PromptStruct, **kwargs) -> str:
        prompt = self._process_prompt(prompt)
        import uuid

        uid = str(uuid.uuid4())
        prefix = f"{uid}->"

        while True:
            _, wlist, _ = select.select([], [self.process.stdin], [], 60)
            if wlist:
                prompt["kwargs"] = kwargs
                self.process.stdin.write(f"{prefix}{json.dumps(prompt)}\n")
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
                            return result[len(prefix) :]
                        elif result.startswith("Error:"):
                            raise RuntimeError("WhisperModel failed: {}".format(result))
                        else:
                            logger.info(result)
                    elif stream == self.process.stderr:
                        err = self.process.stderr.readline().strip()
                        if err:
                            logger.error(f"Process stderr: {err}")
            except BlockingIOError as e:
                logger.error(f"BlockingIOError occurred: {str(e)}")


@isolated("audio_evals/lib/whisper/seed_tts_eval.py")
class SeedTTSWhisperModel(OfflineModel):
    def __init__(
        self,
        path: str = "openai/whisper-large-v3",
        chunk_size: int = 0,
        sample_params: Dict[str, any] = None,
    ):
        if path.startswith("openai/") and not os.path.exists(path):
            path = self._download_model(path)

        self.command_args = {
            "path": path,
            "chunk_size": chunk_size,
        }
        super().__init__(is_chat=True, sample_params=sample_params)

    def _process_prompt(self, prompt: PromptStruct) -> Dict[str, str]:
        if isinstance(prompt, list):
            for content in prompt:
                for line in content["contents"]:
                    if line["type"] == "audio":
                        if not os.path.exists(line["value"]):
                            raise FileNotFoundError(
                                f"Audio file not found: {line['value']}"
                            )
                        return {"audio": line["value"]}
        return prompt

    def _inference(self, prompt: PromptStruct, **kwargs) -> str:
        prompt = self._process_prompt(prompt)
        import uuid

        uid = str(uuid.uuid4())
        prefix = f"{uid}->"

        while True:
            _, wlist, _ = select.select([], [self.process.stdin], [], 60)
            if wlist:
                prompt["kwargs"] = kwargs
                self.process.stdin.write(f"{prefix}{json.dumps(prompt)}\n")
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
                            return result[len(prefix) :]
                        elif result.startswith("Error:"):
                            raise RuntimeError("WhisperModel failed: {}".format(result))
                        else:
                            logger.info(result)
                    elif stream == self.process.stderr:
                        err = self.process.stderr.readline().strip()
                        if err:
                            logger.error(f"Process stderr: {err}")
            except BlockingIOError as e:
                logger.error(f"BlockingIOError occurred: {str(e)}")


@isolated("audio_evals/lib/whisper/cv3.py")
class CV3WhisperModel(OfflineModel):
    def __init__(
        self,
        path: str = "large-v3",
        chunk_size: int = 0,
        sample_params: Dict[str, any] = None,
    ):
        self.command_args = {
            "path": path,
        }
        super().__init__(is_chat=True, sample_params=sample_params)

    def _process_prompt(self, prompt: PromptStruct) -> Dict[str, str]:
        if isinstance(prompt, list):
            for content in prompt:
                for line in content["contents"]:
                    if line["type"] == "audio":
                        if not os.path.exists(line["value"]):
                            raise FileNotFoundError(
                                f"Audio file not found: {line['value']}"
                            )
                        return {"audio": line["value"]}
        return prompt

    def _inference(self, prompt: PromptStruct, **kwargs) -> str:
        prompt = self._process_prompt(prompt)
        import uuid

        uid = str(uuid.uuid4())
        prefix = f"{uid}->"

        while True:
            _, wlist, _ = select.select([], [self.process.stdin], [], 60)
            if wlist:
                prompt["kwargs"] = kwargs
                self.process.stdin.write(f"{prefix}{json.dumps(prompt)}\n")
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
                            return result[len(prefix) :]
                        elif result.startswith("Error:"):
                            raise RuntimeError("WhisperModel failed: {}".format(result))
                        else:
                            logger.info(result)
                    elif stream == self.process.stderr:
                        err = self.process.stderr.readline().strip()
                        if err:
                            logger.error(f"Process stderr: {err}")
            except BlockingIOError as e:
                logger.error(f"BlockingIOError occurred: {str(e)}")
