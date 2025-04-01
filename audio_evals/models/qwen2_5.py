import json
import logging
from typing import Dict
from audio_evals.base import PromptStruct
from audio_evals.models.model import OfflineModel
from audio_evals.isolate import isolated
import select

logger = logging.getLogger(__name__)


@isolated("audio_evals/lib/qwen2-5omni/main.py")
class QwenOmni(OfflineModel):
    def __init__(
        self,
        path: str,
        speech: bool = False,
        sample_params: Dict = None,
        *args,
        **kwargs,
    ):
        self.command_args = {
            "path": path,
        }
        if speech:
            self.command_args["speech"] = True
        super().__init__(is_chat=True, sample_params=sample_params)

    def _parse_content(self, content: Dict):
        assert "type" in content
        return {"type": content["type"], content["type"]: content["value"]}

    def _parse_role_content(self, role_content: Dict):
        for k in ["contents"]:
            if isinstance(role_content[k], list):
                role_content["content"] = [
                    self._parse_content(item) for item in role_content.pop(k)
                ]
            else:
                role_content["content"] = role_content.pop(k)
        return role_content

    def _inference(self, prompt: PromptStruct, **kwargs):
        conversation = [self._parse_role_content(item) for item in prompt]
        if conversation[0]["role"] != "system":
            conversation = [
                {
                    "role": "system",
                    "content": "You are Qwen, a virtual human developed by the Qwen Team, "
                    "Alibaba Group, capable of perceiving auditory and visual "
                    "inputs, as well as generating text and speech",
                }
            ] + conversation

        self.process.stdin.write(f"{json.dumps(conversation)}\n")
        self.process.stdin.flush()

        while True:
            reads, _, _ = select.select(
                [self.process.stdout, self.process.stderr], [], [], 1.0
            )
            for read in reads:
                if read is self.process.stdout:
                    result = self.process.stdout.readline()
                    if result:
                        if result.startswith("Result:"):
                            res = json.loads(result[7:])
                            res["text"] = res["text"].split("assistant")[1].strip()
                            if len(res) == 1:
                                return res["text"]
                            return res
                        elif result.startswith("Error:"):
                            raise RuntimeError("qwen2.5omni failed: {}".format(result))
                        else:
                            logger.info(result)
                if read is self.process.stderr:
                    error_output = self.process.stderr.readline()
                    if error_output:
                        print(f"stderr: {error_output.strip()}")
