import base64
import json
import subprocess
import tempfile
from typing import Dict
from tencentcloud.common import credential
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
from tencentcloud.asr.v20190614 import asr_client, models
from audio_evals.base import PromptStruct
from audio_evals.models.model import APIModel
import logging


logger = logging.getLogger(__name__)


class TencentASRModel(APIModel):
    def __init__(
        self,
        secret_id: str,
        secret_key: str,
        region: str = "ap-guangzhou",
        sample_params: Dict[str, any] = None,
    ):
        super().__init__(False, sample_params)
        self.secret_id = secret_id
        self.secret_key = secret_key
        self.region = region

        # 初始化认证对象
        self.cred = credential.Credential(self.secret_id, self.secret_key)

        # 配置 HTTP 选项
        self.http_profile = HttpProfile()
        self.http_profile.endpoint = "asr.tencentcloudapi.com"

        # 配置客户端参数
        self.client_profile = ClientProfile()
        self.client_profile.httpProfile = self.http_profile

        # 初始化 ASR 客户端
        self.client = asr_client.AsrClient(self.cred, self.region, self.client_profile)

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
            # 读取音频文件并进行 base64 编码
            with open(audio_path, "rb") as f:
                audio_data = f.read()
                audio_base64 = base64.b64encode(audio_data).decode("utf-8")

            # 创建请求对象
            req = models.SentenceRecognitionRequest()
            params = {
                "ProjectId": 0,
                "SubServiceType": 2,
                "SourceType": 1,
                "VoiceFormat": "wav",
                "UsrAudioKey": "session-123",
                "Data": audio_base64,
                "DataLen": len(audio_data),
                **kwargs,
            }
            req.from_json_string(json.dumps(params))

            # 发送请求并获取响应
            resp = self.client.SentenceRecognition(req)
            return resp.Result
