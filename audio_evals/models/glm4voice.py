import json
import tempfile
from typing import Dict

import requests

from audio_evals.base import PromptStruct
from audio_evals.models.model import APIModel
from audio_evals.utils import get_base64_from_file
import soundfile as sf
import numpy as np


def save_audio_response(response, output_file, sample_rate, volume=1.0):
    """保存服务器返回的音频流为文件"""
    if response.status_code == 200:
        text = ""
        audio_tensor = []
        for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
            if chunk:
                data = json.loads(chunk.decode())
                text = data["text"]
                token_id = data['audio']
                if "sampleRate" in data:
                    sample_rate = data["sampleRate"]
                audio_tensor.append(np.array(token_id))
        audio_tensor = np.concatenate(audio_tensor, axis=0)
        audio_tensor *= int(volume)
        audio_tensor = np.array(audio_tensor, dtype=np.int16)
        sf.write(output_file, audio_tensor, sample_rate,)
        return output_file, text
    else:
        raise Exception(f"下载失败，状态码: {response.status_code}")


class GLM4Voice(APIModel):
    def __init__(
        self, url: str, sr: int, volume: float = 1.0, sample_params: Dict[str, any] = None
    ):
        super().__init__(True, sample_params)
        self.url = url
        self.sr = sr
        self.volume = volume

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
            'prompt': '',
            'audio': audio_base64
        }
        response = requests.post(self.url, headers=headers, data=json.dumps(data), stream=True)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            audio, text = save_audio_response(response, f.name, self.sr, self.volume)
            return json.dumps({"audio": audio, "text": text}, ensure_ascii=False)
