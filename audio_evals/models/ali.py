import os
from copy import deepcopy
from http import HTTPStatus
from typing import Dict

from dashscope import MultiModalConversation

from audio_evals.base import PromptStruct
from audio_evals.models.model import APIModel


class AliApi(APIModel):

    def __init__(
        self,
        model_name: str = "qwen2-audio-instruct",
        sample_params: Dict[str, any] = None,
    ):
        super().__init__(True, sample_params)
        self.model = model_name
        assert "DASHSCOPE_API_KEY" in os.environ, ValueError(
            "not found DASHSCOPE_API_KEY in your ENV"
        )

    def _inference(self, prompt: PromptStruct, **kwargs):
        messages = []
        for content in deepcopy(prompt):
            for i, line in enumerate(content["contents"]):
                if line["type"] == "text":
                    content["contents"][i] = {"text": line["value"]}
                else:
                    content["contents"][i] = {
                        line["type"]: "file://{}".format(line["value"])
                    }

            content["content"] = content["contents"]
            del content["contents"]
            messages.append(content)

        response = MultiModalConversation.call(model=self.model, messages=messages)
        if response.status_code == HTTPStatus.OK:
            return response.output.choices[0].message.content[0]["text"]
        raise Exception("{}: {}".format(response.code, response.message))
