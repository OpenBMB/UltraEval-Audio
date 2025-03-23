import logging
from typing import Dict
from audio_evals.base import PromptStruct
from audio_evals.models.model import OfflineModel
from audio_evals.isolate import isolated


logger = logging.getLogger(__name__)


@isolated("audio_evals/lib/utmos/main.py", pre_command="pip install pip==24.0")
class UTMOS(OfflineModel):
    def __init__(self, path: str, sample_params: Dict = None, *args, **kwargs):
        self.command_args = {
            "path": path,
        }
        super().__init__(is_chat=False, sample_params=sample_params)

    def _inference(self, prompt: PromptStruct, **kwargs) -> float:
        self.process.stdin.write(f"{prompt}\n")
        self.process.stdin.flush()
        import select

        while True:
            reads, _, _ = select.select(
                [self.process.stdout, self.process.stderr], [], [], 1.0
            )
            for read in reads:
                if read is self.process.stdout:
                    result = self.process.stdout.readline()
                    if result:
                        if result.startswith("Result:"):
                            return float(result[7:])
                        elif result.startswith("Error:"):
                            raise RuntimeError("utmos failed: {}".format(result))
                        else:
                            logger.info(result)
                if read is self.process.stderr:
                    error_output = self.process.stderr.readline()
                    if error_output:
                        print(f"stderr: {error_output.strip()}")
