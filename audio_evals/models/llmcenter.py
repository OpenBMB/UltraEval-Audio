import os
import random
import time
from typing import Dict

import requests

from audio_evals.base import EarlyStop
from audio_evals.models.model import APIModel, PromptStruct
from audio_evals.utils import convbase64, retry


class LlmCenterModel(APIModel):
    app_code = os.environ.get("LLMCenterAppCode", "interface_evaluation")
    assert "LLMCenterUserToken" in os.environ, ValueError(
        "not found LLMCenterUserToken in your ENV"
    )
    user_token = os.environ["LLMCenterUserToken"]
    access_token = ""
    timeout = 3600 * 24 * 7
    exp_time = 0

    def __init__(self, is_chat, model_id: int, sample_params: Dict[str, any] = None):
        super().__init__(is_chat, sample_params)
        self.model_id = model_id

    @classmethod
    @retry(max_retries=3)
    def get_access_token(cls):
        if cls.access_token and time.time() < cls.exp_time:
            return cls.access_token
        payload = {
            "appCode": cls.app_code,
            "userToken": cls.user_token,
            "expTime": str(cls.timeout),
        }
        url = "https://llm-center.modelbest.cn/llm/client/token/access_token"
        response = requests.get(url, params=payload)
        if response.status_code == 200:
            # 解析返回的JSON数据
            msg_data = response.json()
            cls.access_token = msg_data["data"]
            cls.exp_time = time.time() + cls.timeout - 10
            return cls.access_token
        else:
            RuntimeError(f"get_token failed，状态码: {response.status_code}")

    def _inference(self, prompt: PromptStruct, **kwargs) -> str:
        if isinstance(prompt, str):
            prompt = [
                {"role": "USER", "contents": [{"type": "TEXT", "pairs": prompt}]}
            ]  # the model must be chat type
        else:
            prompt = [
                {
                    "role": item["role"].upper(),
                    "contents": [
                        (
                            {"type": "TEXT", "pairs": item["content"]}
                            if "content" in item
                            else {
                                "type": detail["type"].upper(),
                                "pairs": detail["value"],
                            }
                        )
                        for detail in item["contents"]
                    ],
                }
                for item in prompt
            ]

        for turn in prompt:
            for item in turn["contents"]:
                if item["type"] != "TEXT":
                    item["pairs"] = convbase64(item["pairs"])  # file

        access_token = self.get_access_token()
        payload = {
            "modelId": self.model_id,
            "userSafe": 0,
            "aiSafe": 0,
            "chatMessage": prompt,
            "object_json": "{}",
            "modelParamConfig": {} if kwargs is None else kwargs,
        }
        headers = {
            "app-token": access_token,
            "app-code": "interface_evaluation",
            "Content-Type": "application/json",
        }
        url = "https://llm-center.modelbest.cn/llm/client/conv/accessLargeModel/sync"
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 200:
            msg_data = response.json()
        else:
            raise RuntimeError(
                f"请求失败，状态码: {response.status_code}, {response.text}"
            )

        if msg_data["code"] in {
            102501,  # 超过模型输入限制
            102503,  # 模型内部异常
            102502,  # 模型安审不通过,
            102505,  # 模型输出内容为空
            102504,  # 模型输出大于主动超时时间
            102506,  # 模型服务商网络链接异常
        }:
            raise EarlyStop(f"### 请求失败不用重试 msg_data: {msg_data}")
        if msg_data["code"] == 102429:
            time.sleep(random.uniform(0, 2))
            raise RuntimeError(f"### 请求失败，重试 msg_data: {msg_data}")
        assert 0 == msg_data["code"], msg_data
        content = msg_data["data"]["messages"][0]["content"]
        return content
