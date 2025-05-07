import json
import logging
import select
import subprocess
from typing import Dict

from audio_evals.base import PromptStruct
from audio_evals.models.model import Model, OfflineModel
from audio_evals.isolate import isolated


logger = logging.getLogger(__name__)


class SenseVoice(Model):
    def __init__(
        self,
        path: str,
        cli: str,
        sample_params: Dict[str, any] = None,
    ):
        super().__init__(True, sample_params)
        if not path.endswith("/"):
            path += "/"
        self.path = path
        self.cli = cli

    def _inference(self, prompt: PromptStruct, **kwargs) -> str:
        audio = prompt["audio"]

        cmd = (
            "cd /DATA/disk1/home/shiqundong/project/sensevoice/ &&"
            ". env/bin/activate &&"
            "export LD_LIBRARY_PATH=env/lib/python3.10/site-packages/nvidia/nvjitlink/lib &&"
            "python3 test.py {}".format(audio)
        )

        logger.debug(f"Executing command: {cmd}")
        result = subprocess.run(cmd, capture_output=True, text=True, shell=True)

        if result.returncode == 0:
            text = result.stdout.strip()
            return text
        else:
            raise ValueError(f"Error executing command: {result.stderr}")


@isolated("audio_evals/lib/SenseVoice/main.py")
class SenseVoiceM(OfflineModel):
    def __init__(
        self,
        path: str,
        sample_params: Dict[str, any] = None,
    ):
        self.command_args = {
            "path": path,
        }
        super().__init__(is_chat=True, sample_params=sample_params)

    def _inference(self, prompt: PromptStruct, **kwargs) -> str:
        import uuid

        uid = str(uuid.uuid4())
        prefix = f"{uid}->"

        while True:
            _, wlist, _ = select.select([], [self.process.stdin], [], 60)
            if wlist:
                self.process.stdin.write(f"{prefix}{json.dumps(prompt)}\n")
                self.process.stdin.flush()
                print("already write in")
                break
        while True:
            rlist, _, _ = select.select(
                [self.process.stdout, self.process.stderr], [], [], 60
            )
            if not rlist:
                err_msg = "Read timeout after 6w0 seconds"
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
                            raise RuntimeError("SenseVoiceM failed: {}".format(result))
                        else:
                            logger.info(result)
                    elif stream == self.process.stderr:
                        err = self.process.stderr.readline().strip()
                        if err:
                            logger.error(f"Process stderr: {err}")
            except BlockingIOError as e:
                logger.error(f"BlockingIOError occurred: {str(e)}")
