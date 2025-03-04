import json
from typing import Dict

import requests

from audio_evals.base import PromptStruct
from audio_evals.models.model import APIModel
from audio_evals.utils import get_base64_from_file


class AsrServer(APIModel):
    def __init__(self, url: str, sample_params: Dict[str, any] = None):
        super().__init__(True, sample_params)
        self.url = url

    def _inference(self, prompt: PromptStruct, **kwargs) -> str:

        audio_file = prompt["audio"]
        audio_base64 = get_base64_from_file(audio_file)
        headers = {"Content-Type": "application/json"}
        data = {"audio": audio_base64}
        response = requests.post(
            self.url, headers=headers, data=json.dumps(data), stream=True
        )
        if response.status_code == 200:
            return response.text
        else:
            raise Exception(f"Error: {response.status_code} - {response.text}")
