import http.client
import json
import subprocess
import tempfile
from typing import Dict
from audio_evals.base import PromptStruct
from audio_evals.models.model import APIModel
import logging

logger = logging.getLogger(__name__)


class AliyunASRModel(APIModel):
    def __init__(self, app_key: str, token: str, sample_params: Dict[str, any] = None):
        super().__init__(False, sample_params)
        self.app_key = app_key
        self.token = token
        self.host = "nls-gateway-cn-beijing.aliyuncs.com"
        self.url = f"https://{self.host}/stream/v1/asr"

    def _inference(self, prompt: PromptStruct, **kwargs) -> str:
        audio = prompt["audio"]
        logger.debug(f"Processing audio file: {audio}")

        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp_file:
            audio_path = tmp_file.name
            subprocess.run(
                ["ffmpeg", "-y", "-i", audio, "-ar", "16000", "-ac", "1", audio_path],
                capture_output=True,
                text=True,
                check=True,
            )
            # 读取音频文件
            with open(audio_path, mode="rb") as f:
                audio_content = f.read()

            # 设置HTTPS请求头部
            http_headers = {
                "X-NLS-Token": self.token,
                "Content-type": "application/octet-stream",
                "Content-Length": len(audio_content),
            }

            # 设置RESTful请求参数
            request = self.url + "?appkey=" + self.app_key
            request += "&format=" + kwargs.get("format", "wav")
            request += "&sample_rate=" + str(kwargs.get("sample_rate", 16000))
            if kwargs.get("enable_punctuation_prediction", True):
                request += "&enable_punctuation_prediction=true"
            if kwargs.get("enable_inverse_text_normalization", True):
                request += "&enable_inverse_text_normalization=true"
            if kwargs.get("enable_voice_detection", False):
                request += "&enable_voice_detection=true"

            logger.debug(f"Request: {request}")

            # 发送请求
            conn = http.client.HTTPSConnection(self.host)
            conn.request(
                method="POST", url=request, body=audio_content, headers=http_headers
            )

            response = conn.getresponse()
            logger.debug(
                f"Response status: {response.status}, reason: {response.reason}"
            )

            body = response.read()
            try:
                body = json.loads(body)
                logger.debug(f"Recognize response: {body}")

                status = body.get("status")
                if status == 20000000:
                    result = body.get("result", "")
                    logger.info(f"Recognize result: {result}")
                    return result
                else:
                    logger.error("Recognizer failed!")
                    return "Recognizer failed!"
            except ValueError:
                logger.error("The response is not a JSON format string")
                return "The response is not a JSON format string"
            finally:
                conn.close()
