import json
import tempfile
from typing import Dict

import requests

from audio_evals.base import PromptStruct
from audio_evals.models.model import APIModel
from audio_evals.utils import get_base64_from_file
import numpy as np
import soundfile as sf


OUT_CHANNELS = 1


def save_audio_response(pre_prompt, response, output_file):
    """保存服务器返回的音频流为文件"""
    if response.status_code == 200:
        text = ""
        audios = []
        sample_rate = 16000

        for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
            if chunk:
                data = json.loads(chunk.decode())
                text = data["text"][len(pre_prompt):]
                audios.append(np.array(data["audio"]))

        audio_tensor = np.concatenate(audios, axis=0)
        audio_tensor *= 32767
        audio_tensor = np.array(audio_tensor, dtype=np.int16)
        sf.write(output_file, audio_tensor, sample_rate)
        return output_file, text
    else:
        raise Exception(f"下载失败，状态码: {response.status_code}")


class LlamaOmni(APIModel):
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
        prompt = '''<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful language and speech assistant. You are able to understand the speech content that the user provides, and assist the user with a variety of tasks using natural language.<|eot_id|><|start_header_id|>user<|end_header_id|>

<speech>
Please directly answer the questions in the user's speech.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

'''
        data = {
            "model": 'Llama-3.1-8B-Omni',
            'prompt': prompt,
            'audio': audio_base64,
            "temperature": 0,
            "top_p": 0.7,
            "max_new_tokens": 1024,
            "stop": '<|eot_id|>',
        }
        response = requests.post(self.url, headers=headers, data=json.dumps(data), stream=True)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            audio, text = save_audio_response(prompt, response, f.name)
            return json.dumps({"audio": audio, "text": text}, ensure_ascii=False)

