import os
from typing import Dict, Any
from openai import OpenAI, AzureOpenAI
from azure.core.credentials import AzureKeyCredential


from audio_evals.models.model import APIModel
from audio_evals.base import PromptStruct


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_URL = os.getenv("OPENAI_URL", "https://api.openai.com")


class GPT(APIModel):
    def __init__(
        self,
        model_name: str,
        is_azure: bool = False,
        sample_params: Dict[str, Any] = None,
    ):
        super().__init__(True, sample_params)
        self.model_name = model_name
        assert "OPENAI_API_KEY" in os.environ, ValueError(
            "not found OPENAI_API_KEY in your ENV"
        )
        if is_azure:
            key = os.environ["AZURE_OPENAI_KEY"]
            endpoint = os.environ["AZURE_OPENAI_BASE"]
            print(f"Using Azure OpenAI with key {key} and endpoint {endpoint}")
            self.client = AzureOpenAI(
                api_version="2025-03-01-preview", api_key=key, azure_endpoint=endpoint
            )
        else:
            self.client = OpenAI()

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


class AudioTranscribe(GPT):
    """
    This model is used to transcribe audio to text.
    """

    def _inference(self, prompt, **kwargs):
        audio_file = open(prompt["audio"], "rb")
        transcript = self.client.audio.transcriptions.create(
            model=self.model_name, file=audio_file
        )
        return transcript["text"]
