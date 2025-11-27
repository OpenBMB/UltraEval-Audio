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
    script_path="audio_evals/lib/HiggsAudio/main.py",
    pre_command=(
        "mkdir -p ./third_party && [ ! -d './third_party/higgs-audio' ] && git clone https://github.com/boson-ai/higgs-audio.git ./third_party/higgs-audio || true"
    ),
)
class HiggsAudio(OfflineModel):

    def __init__(
        self,
        path: str,
        audio_tokenizer: str = None,
        ras_win_len: int = 7,
        ras_win_max_num_repeat: int = 2,
        use_static_kv_cache: int = 1,
        sample_params: Dict = None,
        *args,
        **kwargs,
    ):
        if not os.path.exists(path):
            path = self._download_model_from_modelscope(path)
        if not os.path.exists(audio_tokenizer):
            audio_tokenizer = self._download_model_from_modelscope(audio_tokenizer)

        self.command_args = {
            "path": path,
            "audio_tokenizer": audio_tokenizer,
            "ras_win_len": ras_win_len,
            "ras_win_max_num_repeat": ras_win_max_num_repeat,
            "use_static_kv_cache": use_static_kv_cache,
        }

        super().__init__(is_chat=True, sample_params=sample_params)

    def _inference(self, prompt: PromptStruct, **kwargs):
        prompt.update(kwargs)
        import uuid

        uid = str(uuid.uuid4())
        prefix = f"{uid}->"

        while True:
            _, wlist, _ = select.select([], [self.process.stdin], [], 60)
            if wlist:
                self.process.stdin.write(f"{prefix}{json.dumps(prompt)}\n")
                self.process.stdin.flush()
                logger.debug("prompt written to HiggsAudio stdin")
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
                            raise RuntimeError(f"HiggsAudio failed: {result}")
                        else:
                            logger.info(result)
                    elif stream == self.process.stderr:
                        err = self.process.stderr.readline().strip()
                        if err:
                            logger.error(f"Process stderr: {err}")
            except BlockingIOError as e:
                logger.error(f"BlockingIOError occurred: {str(e)}")


@isolated(
    script_path="audio_evals/lib/HiggsAudio/vc.py",
    pre_command=(
        "mkdir -p ./third_party && [ ! -d './third_party/higgs-audio' ] && git clone https://github.com/boson-ai/higgs-audio.git ./third_party/higgs-audio || true"
    ),
)
class HiggsAudioVC(OfflineModel):

    def __init__(
        self,
        path: str,
        audio_tokenizer: str = None,
        ras_win_len: int = 7,
        ras_win_max_num_repeat: int = 2,
        use_static_kv_cache: int = 1,
        sample_params: Dict = None,
        *args,
        **kwargs,
    ):
        if not os.path.exists(path):
            path = self._download_model_from_modelscope(path)
        if not os.path.exists(audio_tokenizer):
            audio_tokenizer = self._download_model_from_modelscope(audio_tokenizer)

        self.command_args = {
            "path": path,
            "audio_tokenizer": audio_tokenizer,
            "ras_win_len": ras_win_len,
            "ras_win_max_num_repeat": ras_win_max_num_repeat,
            "use_static_kv_cache": use_static_kv_cache,
        }

        super().__init__(is_chat=True, sample_params=sample_params)

    def _inference(self, prompt: PromptStruct, **kwargs):
        prompt.update(kwargs)
        import uuid

        uid = str(uuid.uuid4())
        prefix = f"{uid}->"

        while True:
            _, wlist, _ = select.select([], [self.process.stdin], [], 60)
            if wlist:
                self.process.stdin.write(f"{prefix}{json.dumps(prompt)}\n")
                self.process.stdin.flush()
                logger.debug("prompt written to HiggsAudio stdin")
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
                            raise RuntimeError(f"HiggsAudio failed: {result}")
                        else:
                            logger.info(result)
                    elif stream == self.process.stderr:
                        err = self.process.stderr.readline().strip()
                        if err:
                            logger.error(f"Process stderr: {err}")
            except BlockingIOError as e:
                logger.error(f"BlockingIOError occurred: {str(e)}")
