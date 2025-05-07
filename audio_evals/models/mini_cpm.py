import json
from typing import Dict
from audio_evals.base import PromptStruct
from audio_evals.models.model import OfflineModel
from audio_evals.isolate import isolated
import select
import logging


logger = logging.getLogger(__name__)


@isolated("audio_evals/lib/minicpm/main.py")
class MiniCPMo(OfflineModel):
    def __init__(
        self,
        path: str,
        speech: bool = False,
        sample_params: Dict = None,
        *args,
        **kwargs,
    ):
        super().__init__(True, sample_params)
        self.command_args = {
            "path": path,
            "speech": speech,
        }
        super().__init__(is_chat=True, sample_params=sample_params)

    def _inference(self, prompt: PromptStruct, **kwargs) -> str:
        import uuid

        uid = str(uuid.uuid4())
        prefix = f"{uid}->"

        input_o = {"prompt": prompt}
        input_o.update(kwargs)

        while True:
            _, wlist, _ = select.select([], [self.process.stdin], [], 60)
            if wlist:
                self.process.stdin.write(f"{prefix}{json.dumps(input_o)}\n")
                self.process.stdin.flush()
                print("already write in")
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
                            res = json.loads(result[len(prefix) :])
                            if len(res) == 1:
                                return res["text"]
                            return json.dumps(res, ensure_ascii=False)
                        elif result.startswith("Error:"):
                            raise RuntimeError(
                                "mimicpm-o 2.6 failed: {}".format(result)
                            )
                        else:
                            logger.info(result)
                if read is self.process.stderr:
                    error_output = self.process.stderr.readline()
                    if error_output:
                        print(f"stderr: {error_output.strip()}")
