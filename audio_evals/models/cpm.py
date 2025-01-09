import json
import tempfile
from typing import Dict

import requests

from audio_evals.base import PromptStruct
from audio_evals.models.model import APIModel
from audio_evals.utils import get_base64_from_file, decode_base64_to_file
import numpy as np
import soundfile as sf


OUT_CHANNELS = 1


def save_audio_response(response, output_file):
    """保存服务器返回的音频流为文件"""
    if response.status_code == 200:
        text = ""
        audios = ""

        for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
            if chunk:
                data = json.loads(chunk.decode())
                if data["error_code"] == 1:
                    raise Exception(f"service: {data['error']}")
                text += data["text"]
                if data.get("audio", None):
                    audios += data["audio"]

        if audios:
            decode_base64_to_file(audios, output_file)
        return output_file, text
    else:
        raise Exception(f"下载失败，状态码: {response.status_code}")


class CPM3o(APIModel):
    def __init__(
        self, url: str, sample_params: Dict[str, any] = None
    ):
        super().__init__(True, sample_params)
        self.url = url

    def _inference(self, prompt: PromptStruct, **kwargs) -> str:

        audio_file = ""
        for content in prompt:
            if content["role"] == "user":
                for line in content["contents"]:
                    if line["type"] == "audio":
                        audio_file = line["value"]
                        break

        audio_base64 = get_base64_from_file(audio_file)
        headers = {
            'Content-Type': 'application/json'
        }
        data = {
            'audio': audio_base64,
            **kwargs
        }
        response = requests.post(self.url, headers=headers, data=json.dumps(data), stream=True)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            audio, text = save_audio_response(response, f.name)
            return json.dumps({"audio": audio, "text": text}, ensure_ascii=False)


class CPM3oAudio(APIModel):
    def __init__(
        self, url: str, sample_params: Dict[str, any] = None
    ):
        super().__init__(True, sample_params)
        self.url = url

    def _inference(self, prompt: PromptStruct, **kwargs) -> str:

        audio_file = ""
        text = ""
        for content in prompt:
            if content["role"] == "user":
                for line in content["contents"]:
                    if line["type"] == "audio":
                        audio_file = line["value"]
                    elif line["type"] == "text":
                        text = line["value"]

        audio_base64 = get_base64_from_file(audio_file)
        headers = {
            'Content-Type': 'application/json'
        }
        data = {
            'audio': audio_base64,
            'text': text,
            **kwargs
        }
        response = requests.post(self.url, headers=headers, data=json.dumps(data), stream=True)
        audio, text = save_audio_response(response, 'temp')
        return text

