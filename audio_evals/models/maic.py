import json
from typing import Dict

import requests

from audio_evals.base import PromptStruct
from audio_evals.models.model import APIModel
from audio_evals.utils import get_base64_from_file


class MAIC(APIModel):
    def __init__(self, url: str, sample_params: Dict[str, any] = None):
        super().__init__(True, sample_params)
        self.url = url

    def _inference(self, prompt: PromptStruct, **kwargs) -> str:

        audio_file, text = "", ""
        for content in prompt:
            if content["role"] == "user":
                for line in content["contents"]:
                    if line["type"] == "audio":
                        audio_file = line["value"]
                    elif line["type"] == "text":
                        text = line["value"]

        audio_base64 = get_base64_from_file(audio_file) if audio_file else ""
        headers = {"Content-Type": "application/json"}
        data = {"audio": audio_base64, "text": text}
        response = requests.post(self.url, headers=headers, data=json.dumps(data))
        if response.status_code == 200:
            msg_data = response.json()
        else:
            raise RuntimeError(
                f"请求失败，状态码: {response.status_code}, {response.text}"
            )
        return msg_data["text"].replace("</s>", "")
