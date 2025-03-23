import base64
import json
import tempfile
from typing import Dict

import requests

from audio_evals.base import PromptStruct
from audio_evals.models.model import APIModel
from audio_evals.utils import get_base64_from_file


def save_audio_response(response, output_file):
    """保存服务器返回的音频流为文件"""
    if response.status_code == 200:
        audio_data = b""
        text_response = ""

        for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
            if chunk:
                try:
                    event = json.loads(chunk.decode("utf-8"))
                    text_response += event.get("text", "")
                    if output_file:
                        audio_data += base64.b64decode(event.get("audio", ""))
                except json.JSONDecodeError:
                    continue
        if output_file:
            with open(output_file, "wb") as f:
                f.write(audio_data)
        return output_file, text_response
    else:
        raise Exception(f"下载失败，状态码: {response.status_code}")


class StepAudioChat(APIModel):
    def __init__(self, url: str, s2t: bool, sample_params: Dict[str, any] = None):
        super().__init__(True, sample_params)
        self.url = url
        self.s2t = s2t

    def _inference(self, prompt: PromptStruct, **kwargs) -> str:
        text, audio_file = "", ""
        for content in prompt:
            if content["role"] == "user":
                for line in content["contents"]:
                    if line["type"] == "audio":
                        audio_file = line["value"]
                    if line["type"] == "text":
                        text = line["value"]
        endfix = audio_file.split(".")[-1]
        audio_base64 = get_base64_from_file(audio_file)
        headers = {"Content-Type": "application/json"}
        data = {
            "text": text,
            "audio": audio_base64,
            "audio_format": endfix,
        }
        response = requests.post(
            self.url, headers=headers, data=json.dumps(data), stream=True
        )
        if self.s2t:
            _, text = save_audio_response(response, None)
            return text

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            audio, text = save_audio_response(response, f.name)
            return json.dumps({"audio": audio, "text": text}, ensure_ascii=False)
