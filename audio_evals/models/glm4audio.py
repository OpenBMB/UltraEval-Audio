import json
import os
import subprocess
import tempfile
from typing import Dict

import requests

from audio_evals.base import PromptStruct
from audio_evals.models.model import APIModel
from audio_evals.utils import get_base64_from_file
import soundfile as sf
import numpy as np
from pydub import AudioSegment
from pydub.silence import detect_silence


def cut_moshi_greetings(audio_file, output_file):
    # 读取音频文件
    audio = AudioSegment.from_file(audio_file)  # 替换为你的音频文件路径

    # 检测沉默部分
    silence_parts = detect_silence(audio, min_silence_len=1000, silence_thresh=-40)

    # 打印沉默部分
    print("检测到的沉默部分（单位：毫秒）：", audio_file, silence_parts)

    # 如果有沉默部分，将其剪裁掉
    if silence_parts:
        non_silent_audio = audio[:1]  # 删除开场白
        # 大于1s的沉默部分，用1s填充
        for i in range(1, len(silence_parts)):
            non_silent_audio += AudioSegment.silent(1000)
            non_silent_audio += audio[silence_parts[i - 1][1] : silence_parts[i][0]]
        non_silent_audio += AudioSegment.silent(1000)
        non_silent_audio += audio[silence_parts[-1][1] :]  # 保留最后一个沉默后部分
    else:
        non_silent_audio = audio  # 如果没有检测到沉默，保留原始音频

    # 保存处理后的音频
    assert output_file.endswith(".wav"), "输出文件名必须以.wav结尾"
    non_silent_audio.export(output_file, format="wav")  # 替换为目标文件名和格式


def save_audio_response(
    response,
):
    """保存服务器返回的音频流为文件"""
    if response.status_code == 200:
        text = ""
        audio_tensor = []
        for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
            if chunk:
                data = json.loads(chunk.decode())
                text += data["text"]
        return text
    else:
        raise Exception(f"下载失败，状态码: {response.status_code}")


class GLM4Voice(APIModel):
    def __init__(
        self, url: str, cut_greeting: bool = False, sample_params: Dict[str, any] = None
    ):
        super().__init__(True, sample_params)
        self.url = url
        self.cut_greeting = cut_greeting

    def _inference(self, prompt: PromptStruct, **kwargs) -> str:

        audio_file = ""
        for content in prompt:
            if content["role"] == "user":
                for line in content["contents"]:
                    if line["type"] == "audio":
                        audio_file = line["value"]
                        break

        _, file_extension = os.path.splitext(audio_file)
        if file_extension not in [".wav"]:
            with tempfile.NamedTemporaryFile(suffix=".wav") as f:
                subprocess.run(
                    ["ffmpeg", "-y", "-i", audio_file, "-ar", "24000", f.name],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                audio_base64 = get_base64_from_file(f.name)
        else:
            audio_base64 = get_base64_from_file(audio_file)
        headers = {"Content-Type": "application/json"}
        data = {"prompt": "", "audio": audio_base64}
        response = requests.post(
            self.url, headers=headers, data=json.dumps(data), stream=True
        )
        text = save_audio_response(response)
        return text
