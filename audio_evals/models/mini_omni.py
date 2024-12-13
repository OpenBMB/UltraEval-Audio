import json
import tempfile
from typing import Dict

import requests

from audio_evals.base import PromptStruct
from audio_evals.models.model import APIModel
from audio_evals.utils import get_base64_from_file
import wave
import numpy as np


OUT_CHANNELS = 1


def save_audio_response(response, output_file):
    """保存服务器返回的音频流为文件"""
    if response.status_code == 200:
        text = ""
        with wave.open(output_file, 'wb') as wf:
            wf.setnchannels(OUT_CHANNELS)
            wf.setsampwidth(2)  # 2 bytes per sample (16-bit audio)
            wf.setframerate(24000)

            for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
                if chunk:
                    data = json.loads(chunk.decode())
                    text = data["text"]
                    audio_data = np.frombuffer(bytes.fromhex(data["audio"]), dtype=np.int16)
                    audio_data = audio_data.reshape(-1, OUT_CHANNELS)
                    wf.writeframes(audio_data.tobytes())
        return output_file, text
    else:
        raise Exception(f"下载失败，状态码: {response.status_code}")


class MiniOmni(APIModel):
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
            'Content-Type': 'application/json',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        data = {
            'audio': audio_base64
        }
        response = requests.post(self.url, headers=headers, data=json.dumps(data), stream=True)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            audio, text = save_audio_response(response, f.name)
            return json.dumps({"audio": audio, "text": text}, ensure_ascii=False)

