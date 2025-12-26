import logging
import select
from audio_evals.base import PromptStruct
from audio_evals.models.model import OfflineModel
from audio_evals.isolate import isolated
import json
import os

logger = logging.getLogger(__name__)


@isolated(
    "audio_evals/lib/CosyVoice/main.py",
    pre_command="pip install tensorrt-cu12==10.0.1 tensorrt-cu12-bindings==10.0.1 tensorrt-cu12-libs==10.0.1",
)
class CosyVoice(OfflineModel):
    def __init__(self, path: str, vc_mode: bool = False, *args, **kwargs):

        if not os.path.exists(path):
            path = self._download_model(path)

        self.command_args = {
            "path": path,
        }
        if vc_mode:
            self.command_args["vc_mode"] = ""

        super().__init__(is_chat=False, sample_params=None)

    def _inference(self, prompt: PromptStruct, **kwargs) -> str:

        import uuid

        uid = str(uuid.uuid4())
        prefix = f"{uid}->"

        while True:
            # Wait for stdin to be writable
            _, wlist, _ = select.select([], [self.process.stdin], [], 180)
            if not wlist:
                raise RuntimeError("Write timeout after 180 seconds")
            try:
                self.process.stdin.write(f"{prefix}{json.dumps(prompt)}\n")
                self.process.stdin.flush()
                break
            except BlockingIOError:
                continue

        while True:
            # Wait for stdout or stderr to be readable
            rlist, _, _ = select.select(
                [self.process.stdout, self.process.stderr], [], [], 180
            )
            if not rlist:
                err_msg = "Read timeout after 180 seconds"
                logger.error(err_msg)
                raise RuntimeError(err_msg)

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
                            raise RuntimeError("CosyVoice failed: {}".format(result))
                        else:
                            logger.info(result)
                    elif stream == self.process.stderr:
                        err = self.process.stderr.readline().strip()
                        if err:
                            logger.error(f"Process stderr: {err}")
                        continue
            except BlockingIOError as e:
                logger.error(f"BlockingIOError occurred: {str(e)}")
                continue


@isolated(
    "audio_evals/lib/CosyVoiceOffical/main.py",
    pre_command="mkdir -p ./third_party && ([ ! -d './third_party/CosyVoice' ] && "
    "git clone https://github.com/FunAudioLLM/CosyVoice.git ./third_party/CosyVoice&& cd ./third_party/CosyVoice&& git submodule update --init --recursive && cd ../../) || true && pwd",
)
class CosyVoiceLast(OfflineModel):
    def __init__(self, path: str, *args, **kwargs):

        if not os.path.exists(path):
            path = self._download_model(path)

        self.command_args = {
            "path": path,
        }

        super().__init__(is_chat=False, sample_params=None)

    def _inference(self, prompt: PromptStruct, **kwargs) -> str:

        import uuid

        uid = str(uuid.uuid4())
        prefix = f"{uid}->"

        while True:
            # Wait for stdin to be writable
            _, wlist, _ = select.select([], [self.process.stdin], [], 60)
            if not wlist:
                raise RuntimeError("Write timeout after 60 seconds")
            try:
                self.process.stdin.write(f"{prefix}{json.dumps(prompt)}\n")
                self.process.stdin.flush()
                break
            except BlockingIOError:
                continue

        while True:
            # Wait for stdout or stderr to be readable
            rlist, _, _ = select.select(
                [self.process.stdout, self.process.stderr], [], [], 180
            )
            if not rlist:
                err_msg = "Read timeout after 180 seconds"
                logger.error(err_msg)
                raise RuntimeError(err_msg)

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
                            raise RuntimeError("CosyVoice failed: {}".format(result))
                        else:
                            logger.info(result)
                    elif stream == self.process.stderr:
                        err = self.process.stderr.readline().strip()
                        if err:
                            logger.error(f"Process stderr: {err}")
                        continue
            except BlockingIOError as e:
                logger.error(f"BlockingIOError occurred: {str(e)}")
                continue
