import logging
import os
from typing import Dict
from audio_evals.base import PromptStruct
from audio_evals.models.model import OfflineModel
from audio_evals.isolate import isolated
import select


logger = logging.getLogger(__name__)


@isolated("audio_evals/lib/paraformer/main.py")
class Paraformer(OfflineModel):
    def __init__(self, path: str, sample_params: Dict = None, *args, **kwargs):
        if not os.path.exists(path):
            path = self._download_model(path)

        self.command_args = {
            "path": path,
        }
        super().__init__(is_chat=False, sample_params=sample_params)

    def _inference(self, prompt: PromptStruct, **kwargs) -> float:
        audio = prompt["audio"]
        import uuid

        uid = str(uuid.uuid4())
        prefix = f"{uid}->"
        while True:
            _, wlist, _ = select.select([], [self.process.stdin], [], 60)
            if wlist:
                self.process.stdin.write(f"{prefix}{audio}\n")
                self.process.stdin.flush()
                print("already write in")
                break
            print("waiting for write")
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
                            raise RuntimeError("FireRedASR failed: {}".format(result))
                        else:
                            logger.info(result)
                if read is self.process.stderr:
                    error_output = self.process.stderr.readline()
                    if error_output:
                        print(f"stderr: {error_output.strip()}")
