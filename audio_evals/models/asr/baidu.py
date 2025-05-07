import time
import requests
import tempfile
import subprocess
from typing import Dict
from audio_evals.base import PromptStruct
from audio_evals.models.model import Model
import logging

from audio_evals.utils import get_base64_from_file

logger = logging.getLogger(__name__)


class BaiduASRModel(Model):
    def __init__(
        self, api_key: str, secret_key: str, sample_params: Dict[str, any] = None
    ):
        super().__init__(True, sample_params)
        self.api_key = api_key
        self.secret_key = secret_key
        self.token = None
        self.token_expire_time = 0
        self._update_token()

    def _update_token(self):
        if self.token and time.time() < self.token_expire_time - 60:
            return
        url = f"https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id={self.api_key}&client_secret={self.secret_key}"
        response = requests.post(url)
        if response.status_code == 200:
            data = response.json()
            self.token = data["access_token"]
            self.token_expire_time = time.time() + data["expires_in"]
            logger.info("Token updated successfully")
        else:
            raise RuntimeError("Failed to get access token")

    def _inference(self, prompt: PromptStruct, **kwargs) -> str:
        self._update_token()

        # 获取音频文件路径
        audio = prompt["audio"]

        # 使用临时文件存储转换后的 WAV 格式音频
        with tempfile.NamedTemporaryFile(suffix=".wav") as wav_file:
            # 使用 ffmpeg 转换音频格式为 WAV
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    audio,
                    "-ar",
                    "16000",
                    "-ac",
                    "1",
                    wav_file.name,
                ],
                capture_output=True,
                text=True,
                check=True,
            )

            # 读取转换后的音频数据
            with open(wav_file.name, "rb") as f:
                audio_data = f.read()

            # 调用百度 ASR API
            url = f"https://vop.baidu.com/server_api"
            headers = {"Content-Type": "application/json"}
            payload = {
                "format": "wav",
                "rate": 16000,
                "channel": 1,
                "cuid": "123456",
                "dev_pid": kwargs["dev_pid"],
                "token": self.token,
                "speech": get_base64_from_file(wav_file.name),
                "len": len(audio_data),
            }
            response = requests.post(url, headers=headers, json=payload)

            if response.status_code == 200:
                result = response.json()
                if result["err_no"] == 0:
                    return result["result"][0]
                else:
                    raise RuntimeError(f"ASR Error: {result}")
            else:
                raise RuntimeError("Failed to get ASR result")
