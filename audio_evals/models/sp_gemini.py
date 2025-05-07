import os
from typing import Dict

from audio_evals.base import PromptStruct, EarlyStop
from audio_evals.models.model import APIModel
from google import genai
from google.genai import types

from audio_evals.utils import MIME_TYPE_MAP


class SPGemini(APIModel):

    def __init__(
        self, model_name: str = "gemini-1.5-flash", sample_params: Dict[str, any] = None
    ):
        super().__init__(True, sample_params)
        self.model = model_name
        self.client = genai.Client(
            vertexai=True, project="amp-pwa-268810", location="us-central1"
        )

    def _inference(self, prompt: PromptStruct, **kwargs) -> str:
        audio, text = None, None
        for line in prompt[0]["contents"]:
            if line["type"] == "audio":
                audio = line["value"]
            if line["type"] == "text":
                text = line["value"]
        assert os.path.exists(audio), EarlyStop(f"not found audio file: {audio}")
        file_extension = os.path.splitext(audio)[1].lower()

        audio_url = audio.replace(
            "/tmp/jfs-training-root/training-root/user/hf-download/AudioEvals/",
            "https://hf-download-data.ks3-cn-beijing.ksyuncs.com/AudioEvals/",
        )

        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part(
                        text=text,
                    ),
                    types.Part(
                        file_data=types.FileData(
                            file_uri=audio_url,
                            mime_type=MIME_TYPE_MAP[file_extension],
                        ),
                    ),
                ],
            )
        ]
        generate_content_config = types.GenerateContentConfig(
            max_output_tokens=8192,
            response_modalities=["TEXT"],
            safety_settings=[
                types.SafetySetting(
                    category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_HARASSMENT", threshold="OFF"
                ),
            ],
        )

        res = ""
        for chunk in self.client.models.generate_content_stream(
            model=self.model,
            contents=contents,
            config=generate_content_config,
        ):
            res += chunk.text
        return res
