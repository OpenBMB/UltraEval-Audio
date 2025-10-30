from itertools import chain
import json
import logging
import os
from typing import Dict
from audio_evals.base import PromptStruct
from audio_evals.models.model import OfflineModel
from audio_evals.isolate import isolated
import select

logger = logging.getLogger(__name__)


@isolated("audio_evals/lib/Kimi-Audio/main.py")
class KimiAudioModel(OfflineModel):
    def __init__(
        self,
        model_path: str = "moonshotai/Kimi-Audio-7B-Instruct",
        speech: bool = False,
        sample_params: Dict = None,
        *args,
        **kwargs,
    ):
        if model_path == "moonshotai/Kimi-Audio-7B-Instruct" and not os.path.exists(
            model_path
        ):
            model_path = self._download_model(model_path)

        self.command_args = {
            "model_path": model_path,
        }
        if speech:
            self.command_args["speech"] = ""
        super().__init__(is_chat=True, sample_params=sample_params)

    def _parse_role_content(self, role_content: Dict):
        assert isinstance(
            role_content["contents"], list
        ), "prompt should be list not string"

        res = []

        for c in role_content["contents"]:
            temp = {
                "role": role_content["role"],
                "message_type": c["type"],
                "content": c["value"],
            }
            res.append(temp)
        return res

    def _inference(self, prompt: PromptStruct, **kwargs):
        valid_propmt = list(chain(*[self._parse_role_content(item) for item in prompt]))

        import uuid

        uid = str(uuid.uuid4())
        prefix = f"{uid}->"
        payload = {"messages": valid_propmt}
        while True:
            _, wlist, _ = select.select([], [self.process.stdin], [], 60)
            if wlist:
                self.process.stdin.write(f"{prefix}{json.dumps(payload)}\n")
                self.process.stdin.flush()
                break
        while True:
            rlist, _, _ = select.select(
                [self.process.stdout, self.process.stderr], [], [], 60
            )
            if not rlist:
                err_msg = "Read timeout after 60 seconds"
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
                            res = json.loads(result[len(prefix) :])
                            if len(res) == 1:
                                return res["text"]
                            return result[len(prefix) :]
                        elif result.startswith("Error:"):
                            raise RuntimeError("Kimi-Audio failed: {}".format(result))
                        else:
                            logger.info(result)
                    elif stream == self.process.stderr:
                        err = self.process.stderr.readline().strip()
                        if err:
                            logger.error(f"Process stderr: {err}")
            except BlockingIOError as e:
                logger.error(f"BlockingIOError occurred: {str(e)}")
