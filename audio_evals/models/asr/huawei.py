import os
import json
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


class Huawei(Model):
    def __init__(
        self,
        username,
        password,
        domain_name,
        project_name,
        region,
        project_id,
        sample_params: Dict[str, any] = None,
    ):
        super().__init__(True, sample_params)
        self.username = username
        self.password = password
        self.domain_name = domain_name
        self.project_name = project_name
        self.region = region
        self.iam_url = f"https://iam.{region}.myhuaweicloud.com/v3/auth/tokens"
        self.asr_url = f"https://sis-ext.{region}.myhuaweicloud.com/v1"
        self.token = None
        self.token_expiry = 0
        self.project_id = project_id

    def _get_auth_token(self) -> str:
        if self.token and time.time() < self.token_expiry:
            return self.token  # Token 仍然有效

        payload = {
            "auth": {
                "identity": {
                    "methods": ["password"],
                    "password": {
                        "user": {
                            "name": self.username,
                            "password": self.password,
                            "domain": {"name": self.domain_name},
                        }
                    },
                },
                "scope": {"project": {"name": self.project_name}},
            }
        }

        headers = {"Content-Type": "application/json"}
        response = requests.post(self.iam_url, json=payload, headers=headers)
        if response.status_code == 201:
            self.token = response.headers["X-Subject-Token"]
            self.token_expiry = time.time() + 23 * 3600  # 提前 1 小时过期防止 API 拒绝
            return self.token
        else:
            raise Exception(f"Failed to get token: {response.text}")

    def _inference(self, prompt: PromptStruct, **kwargs) -> str:
        audio = prompt["audio"]
        with tempfile.NamedTemporaryFile(suffix=".wav") as wav_file:
            # 转换音频格式为wav
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

            # 获取认证token
            token = self._get_auth_token()

            headers = {"Content-Type": "application/json", "X-Auth-Token": token}

            url = f"{self.asr_url}/{self.project_id}/asr/short-audio"
            payload = {
                "config": {
                    "audio_format": "wav",
                    "property": kwargs["property"],
                    "add_punc": "yes",
                },
                "data": get_base64_from_file(wav_file.name),
            }

            response = requests.post(url, json=payload, headers=headers)
            if response.status_code == 200:
                res = response.json()
                try:
                    return res["result"]["text"]
                except KeyError:
                    raise Exception(f"ASR result is not in expected format: {res}")
            else:
                raise Exception(f"ASR request failed: {response.text}")
