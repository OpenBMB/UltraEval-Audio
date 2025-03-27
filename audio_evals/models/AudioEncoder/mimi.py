import logging
from audio_evals.base import PromptStruct
from audio_evals.models.model import OfflineModel
from audio_evals.isolate import isolated


logger = logging.getLogger(__name__)


@isolated("audio_evals/lib/mimi/main.py")
class MIMI(OfflineModel):
    def __init__(
        self, path: str, mono: bool = False, stereo: bool = False, *args, **kwargs
    ):
        self.command_args = {
            "path": path,
            "mono": mono,
            "stereo": stereo,
        }
        super().__init__(is_chat=False, sample_params=None)

    def _inference(self, prompt: PromptStruct, **kwargs) -> str:
        audio_path = prompt["audio"]
        self.process.stdin.write(f"{audio_path}\n")
        self.process.stdin.flush()
        while True:
            result = self.process.stdout.readline().strip()
            if result.startswith("Result:"):
                return result[7:]
            elif result.startswith("Error:"):
                raise RuntimeError("MIMI failed: {}".format(result))
            else:
                logger.info(result)
