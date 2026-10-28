import base64
import logging
import os
from copy import deepcopy
from typing import Dict

import requests

logger = logging.getLogger(__name__)

from audio_evals.base import PromptStruct
from audio_evals.models.model import APIModel
from audio_evals.utils import MIME_TYPE_MAP


class Gemini(APIModel):

    def __init__(
        self,
        model_name: str = "gemini-1.5-flash",
        key: str = None,
        base_url: str = None,
        sample_params: Dict[str, any] = None,
    ):
        super().__init__(True, sample_params)
        self.model = model_name
        self.key = (
            key
            if key
            else os.environ.get(
                "GOOGLE_API_KEY", "must be set GOOGLE_API_KEY in your ENV"
            )
        )
        self.base_url = (
            base_url
            or os.environ.get(
                "GOOGLE_BASE_URL", "https://generativelanguage.googleapis.com"
            )
        ).rstrip("/")

    def _inference(self, prompt: PromptStruct, **kwargs) -> str:
        system_instruct = ""

        payload = {"contents": []}

        for content in deepcopy(prompt):
            if content["role"] == "user":
                for i, line in enumerate(content["contents"]):
                    if line["type"] == "text":
                        content["contents"][i] = {"text": line["value"]}
                    else:
                        file_path = line["value"]
                        file_extension = os.path.splitext(file_path)[1].lower()
                        if file_extension not in MIME_TYPE_MAP:
                            ValueError(
                                "only support [{}] type, but get {}".format(
                                    MIME_TYPE_MAP.keys(), file_extension
                                )
                            )

                        with open(file_path, "rb") as file:
                            file_content = file.read()
                            base64_encoded = base64.b64encode(file_content).decode(
                                "utf-8"
                            )

                        content["contents"][i] = {
                            "inlineData": {
                                "mimeType": MIME_TYPE_MAP[file_extension],
                                "data": base64_encoded,
                            }
                        }
            elif content["role"] == "system":
                system_instruct = content["content"]
                continue
            else:
                raise ValueError(
                    "only support role [user, system], but get {}".format(
                        content["role"]
                    )
                )

            content["parts"] = content["contents"]
            del content["contents"]
            payload["contents"].append(content)

        inf_args = {}
        if system_instruct:
            inf_args["systemInstruction"] = system_instruct
        if kwargs:
            if "safetySettings" in kwargs:
                payload["safetySettings"] = kwargs.pop("safetySettings")
            payload["generationConfig"] = kwargs

        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer {}".format(self.key),
        }
        url = "{}/v1beta/models/{}:generateContent?key={}".format(
            self.base_url, self.model, self.key
        )

        logger.debug("payload: %s", payload)
        # Always bound the request: a socket that never answers would otherwise
        # wedge the worker forever (the @retry on APIModel.inference only fires
        # on exceptions, and a hang raises nothing).
        response = requests.post(url, json=payload, headers=headers, timeout=(10, 300))
        logger.debug("response: %s", response.text)
        if response.status_code == 200:
            msg_data = response.json()
        else:
            raise RuntimeError(
                f"请求失败，状态码: {response.status_code}, {response.text}"
            )

        candidates = msg_data.get("candidates") or []
        if not candidates:
            # A blocked prompt still answers 200, just without any candidate.
            # Raising keeps the sample out of the metric instead of scoring it
            # as an empty transcription.
            raise RuntimeError(
                "响应中没有 candidates: {}".format(
                    msg_data.get("promptFeedback", msg_data)
                )
            )

        parts = candidates[0].get("content", {}).get("parts") or []
        # Thinking models prepend a `thought: true` part, whose text is empty
        # whenever no thought summary was produced, and the answer lands in a
        # later part. Taking parts[0] blindly would return "". Concatenate the
        # non-thought text parts, the way the genai SDK builds response.text.
        text = "".join(p["text"] for p in parts if "text" in p and not p.get("thought"))
        if not text:
            raise RuntimeError(
                "响应中没有文本内容, finishReason: {}, {}".format(
                    candidates[0].get("finishReason"), msg_data
                )
            )
        return text
