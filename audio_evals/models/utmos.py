import logging
import os
from typing import Dict
from audio_evals.base import PromptStruct
from audio_evals.models.model import OfflineModel
from audio_evals.isolate import isolated
import select


logger = logging.getLogger(__name__)


@isolated(
    "audio_evals/lib/utmos/main.py",
    pre_command="pip install pip==24.0 &&export SACREBLEU_ROOT=envs/utmos/.sacrebleu",
)
class UTMOS(OfflineModel):
    def __init__(
        self,
        path: str = "sarulab-speech/UTMOS-demo",
        sample_params: Dict = None,
        *args,
        **kwargs,
    ):
        if path == "sarulab-speech/UTMOS-demo" and not os.path.exists(path):
            path = self._download_model(path, repo_type="space")

        self.command_args = {
            "path": path,
        }
        super().__init__(is_chat=False, sample_params=sample_params)

    def _inference(self, prompt: PromptStruct, **kwargs) -> float:
        import uuid

        uid = str(uuid.uuid4())
        prefix = f"{uid}->"

        while True:
            _, wlist, _ = select.select([], [self.process.stdin], [], 60)
            if wlist:
                self.process.stdin.write(f"{prefix}{prompt}\n")
                self.process.stdin.flush()
                logger.info("already write in")
                break

        while True:
            reads, _, _ = select.select(
                [self.process.stdout, self.process.stderr], [], [], 1.0
            )
            for read in reads:
                if read is self.process.stdout:
                    result = self.process.stdout.readline()
                    if result:
                        if result.startswith(prefix):
                            self.process.stdin.write("{}close\n".format(prefix))
                            self.process.stdin.flush()
                            return float(result[len(prefix) :])
                        elif result.startswith("Error:"):
                            raise RuntimeError("utmos failed: {}".format(result))
                        else:
                            logger.info(result)
                if read is self.process.stderr:
                    error_output = self.process.stderr.readline()
                    if error_output:
                        print(f"stderr: {error_output.strip()}")
