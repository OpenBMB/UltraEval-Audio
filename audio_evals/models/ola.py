import json
import logging
from typing import Dict
from audio_evals.base import PromptStruct
from audio_evals.models.model import OfflineModel
from audio_evals.isolate import isolated
import select
import uuid


logger = logging.getLogger(__name__)


@isolated("audio_evals/lib/Ola/main.py", pre_command="pip install pip==24.0")
class OlaModel(OfflineModel):
    def __init__(
        self,
        path: str,
        sample_params: Dict = None,
        *args,
        **kwargs,
    ):
        self.command_args = {
            "path": path,
        }
        super().__init__(is_chat=True, sample_params=sample_params)

    def _parse_content(self, content: Dict):
        assert "type" in content
        return {content["type"]: content["value"]}

    def _parse_role_content(self, role_content: Dict):
        res = {}
        for k in ["contents"]:
            if isinstance(role_content[k], list):
                for item in role_content.pop(k):
                    res.update(self._parse_content(item))
        return res

    def _inference(self, prompt: PromptStruct, **kwargs):
        assert (
            len(prompt) == 1
        ), "Only support single turn conversation, but got {}".format(prompt)
        conversation = self._parse_role_content(prompt[0])
        uid = str(uuid.uuid4())
        prefix = f"{uid}->"

        while True:
            _, wlist, _ = select.select([], [self.process.stdin], [], 60)
            if wlist:
                self.process.stdin.write(f"{prefix}{json.dumps(conversation)}\n")
                self.process.stdin.flush()
                print("already write in")
                break
            print("waiting for write")

        while True:
            reads, _, _ = select.select(
                [self.process.stdout, self.process.stderr], [], [], 60
            )
            if not reads:
                err_msg = "Read timeout after 60 seconds"
                logger.error(err_msg)
                raise RuntimeError(err_msg)
            try:
                for stream in reads:
                    if stream == self.process.stdout:
                        result = self.process.stdout.readline().strip()
                        if not result:
                            continue
                        elif result.startswith(prefix):
                            self.process.stdin.write("{}close\n".format(prefix))
                            self.process.stdin.flush()
                            return result[len(prefix) :]
                        elif result.startswith("Error:"):
                            raise RuntimeError("ola failed: {}".format(result))
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
