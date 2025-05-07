import os
from typing import Dict, Any
from openai import OpenAI

from audio_evals.models.model import APIModel
from audio_evals.base import PromptStruct


API_KEY = os.getenv("DOUBAO_API_KEY")
URL = os.getenv("DOUBAO_URL")


class Doubao(APIModel):
    def __init__(
        self, model_name: str, api_key: str = None, sample_params: Dict[str, Any] = None
    ):
        super().__init__(True, sample_params)
        self.model_name = model_name
        assert "DOUBAO_API_KEY" in os.environ or api_key is not None, ValueError(
            "not found DOUBAO_API_KEY in your ENV"
        )
        if api_key is None:
            api_key = os.environ.get("DOUBAO_API_KEY")
        self.client = OpenAI(
            # 此为默认路径，您可根据业务所在地域进行配置
            base_url="https://ark.cn-beijing.volces.com/api/v3",
            # 从环境变量中获取您的 API Key
            api_key=api_key,
        )

    def _inference(self, prompt: PromptStruct, **kwargs) -> str:

        messages = []
        for item in prompt:
            messages.append(
                {"role": item["role"], "content": item["contents"][0]["value"]}
            )

        response = self.client.chat.completions.create(
            model=self.model_name, messages=messages, **kwargs
        )

        return response.choices[0].message.content


class DoubaoAudioPipeline(APIModel):
    def __init__(self, asr: str, llm: str):
        super().__init__(True)
        from audio_evals.registry import registry

        self.asr = registry.get_model(asr)
        self.llm = registry.get_model(llm)

    def _inference(self, prompt: PromptStruct, **kwargs) -> str:
        text = self.asr.inference(prompt)
        res = self.llm.inference(text)
        return res
