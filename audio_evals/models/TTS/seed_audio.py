"""Volcengine Seed Audio (Doubao audio generation 1.0) HTTP API model.

Official documentation:
https://docs.volcengine.com/docs/doubao-voice/audio-generation-http
"""

import base64
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import requests

from audio_evals.base import PromptStruct
from audio_evals.models.model import APIModel


DEFAULT_API_URL = "https://openspeech.bytedance.com/api/v3/tts/create"
# The API rejects longer prompts server-side; fail fast with a clear message.
MAX_TEXT_PROMPT_CHARS = 3000
SUPPORTED_REFERENCE_SUFFIXES = {".wav", ".mp3", ".pcm", ".ogg", ".opus"}
OUTPUT_SUFFIXES = {
    "mp3": ".mp3",
    "ogg_opus": ".ogg",
    "opus": ".opus",
    "pcm": ".pcm",
    "wav": ".wav",
}
# Prompt keys the wrapper consumes itself, beyond the raw request fields.
PROMPT_FIELDS = ("text_prompt", "prompt_audio", "references", "speaker")
# Remaining request-body fields per the API reference. Anything else in the
# rendered prompt is a dataset column that must be rewritten into
# ``text_prompt`` by a prompt from registry/prompt/seed_audio.yaml.
SUPPORTED_REQUEST_FIELDS = frozenset({"model", "audio_config", "watermark"})


class SeedAudio(APIModel):
    """Volcengine Seed Audio HTTP API model.

    The rendered prompt must already speak the Seed Audio request schema:
    - ``text_prompt`` (required): the free-form scene description the API
      expects; reference audio is addressed positionally from inside it as
      ``@音频1``..``@音频3``.
    - ``prompt_audio``: local reference-audio file, base64-inlined into
      ``references[0].audio_data``.
    - ``references``: raw ``references`` entries (``speaker`` / ``audio_data``
      / ``audio_url`` / ``image_data`` / ``image_url``).
    - ``speaker``: Doubao-2.0 or voice-clone speaker ID, sent as
      ``references[0].speaker``.

    Dataset columns such as ``text``, ``instruction`` and ``prompt_text``
    are not request fields; the prompts in registry/prompt/seed_audio.yaml
    rewrite them into ``text_prompt``. Passing them here is an error, as is
    any key outside the documented request body.
    """

    def __init__(
        self,
        model_name: str = "seed-audio-1.0",
        api_key: Optional[str] = None,
        api_url: str = DEFAULT_API_URL,
        request_timeout: int = 300,
        sample_params: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(is_chat=False, sample_params=sample_params)
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("SEED_AUDIO_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Seed Audio API key is required. Set SEED_AUDIO_API_KEY or pass api_key."
            )
        self.api_url = api_url
        self.request_timeout = request_timeout
        self.session = requests.Session()

    @staticmethod
    def _prompt_to_payload(prompt: PromptStruct) -> Dict[str, Any]:
        if isinstance(prompt, dict):
            return dict(prompt)
        if isinstance(prompt, str):
            return {"text_prompt": prompt}
        raise TypeError("Seed Audio expects a string or dictionary prompt.")

    @staticmethod
    def _encode_reference_audio(audio_path: str) -> str:
        path = Path(audio_path)
        if not path.is_file():
            raise FileNotFoundError(f"Reference audio not found: {audio_path}")
        if path.suffix.lower() not in SUPPORTED_REFERENCE_SUFFIXES:
            supported = ", ".join(sorted(SUPPORTED_REFERENCE_SUFFIXES))
            raise ValueError(
                f"Unsupported reference audio format {path.suffix!r}; "
                f"expected one of: {supported}"
            )
        if path.stat().st_size > 10 * 1024 * 1024:
            raise ValueError("Seed Audio reference audio must not exceed 10 MB.")
        return base64.b64encode(path.read_bytes()).decode("ascii")

    def _build_request(
        self, prompt: PromptStruct, inference_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        values = self._prompt_to_payload(prompt)
        values.update(inference_params)

        text_prompt = values.pop("text_prompt", None)
        prompt_audio = values.pop("prompt_audio", None)
        references = values.pop("references", None)
        speaker = values.pop("speaker", None)

        unsupported = sorted(set(values) - SUPPORTED_REQUEST_FIELDS)
        if unsupported:
            raise ValueError(
                "Seed Audio does not accept {} in the prompt. Supported "
                "fields: {}, or the request fields ({}). Rewrite dataset "
                "columns with a prompt from registry/prompt/seed_audio.yaml.".format(
                    ", ".join(unsupported),
                    ", ".join(PROMPT_FIELDS),
                    ", ".join(sorted(SUPPORTED_REQUEST_FIELDS)),
                )
            )

        reference_sources = sorted(
            name
            for name, value in (
                ("prompt_audio", prompt_audio),
                ("references", references),
                ("speaker", speaker),
            )
            if value
        )
        if len(reference_sources) > 1:
            raise ValueError(
                "Use at most one of `prompt_audio`, `references`, `speaker` "
                f"for Seed Audio; got: {', '.join(reference_sources)}."
            )

        if not isinstance(text_prompt, str) or not text_prompt.strip():
            raise ValueError(
                "Seed Audio requires a non-empty `text_prompt`. Use a prompt "
                "from registry/prompt/seed_audio.yaml to build one from "
                "dataset columns."
            )
        if len(text_prompt) > MAX_TEXT_PROMPT_CHARS:
            raise ValueError(
                "text_prompt is {} characters, over the {} character API "
                "limit.".format(len(text_prompt), MAX_TEXT_PROMPT_CHARS)
            )

        request: Dict[str, Any] = {
            "model": self.model_name,
            "text_prompt": text_prompt,
            **values,
        }

        if prompt_audio:
            request["references"] = [
                {"audio_data": self._encode_reference_audio(str(prompt_audio))}
            ]
        elif references:
            request["references"] = references
        elif speaker:
            request["references"] = [{"speaker": speaker}]

        return request

    @staticmethod
    def _output_suffix(request: Dict[str, Any]) -> str:
        audio_config = request.get("audio_config") or {}
        output_format = str(audio_config.get("format", "wav")).lower()
        return OUTPUT_SUFFIXES.get(output_format, f".{output_format}")

    def _write_audio_response(self, response_data: Dict[str, Any], suffix: str) -> str:
        audio_data = response_data.get("audio")
        audio_url = response_data.get("url")

        if audio_data:
            try:
                content = base64.b64decode(audio_data, validate=True)
            except (ValueError, TypeError) as error:
                raise RuntimeError(
                    "Seed Audio returned invalid base64 audio data."
                ) from error
        elif audio_url:
            audio_response = self.session.get(audio_url, timeout=self.request_timeout)
            audio_response.raise_for_status()
            content = audio_response.content
        else:
            raise RuntimeError(
                "Seed Audio response contains neither `audio` nor `url`."
            )

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as output:
            output.write(content)
            return output.name

    def _inference(self, prompt: PromptStruct, **kwargs) -> str:
        request = self._build_request(prompt, kwargs)
        response = self.session.post(
            self.api_url,
            headers={
                "Content-Type": "application/json",
                "X-Api-Key": self.api_key,
                "X-Api-Request-Id": str(uuid.uuid4()),
            },
            json=request,
            timeout=self.request_timeout,
        )

        try:
            response_data = response.json()
        except ValueError as error:
            raise RuntimeError(
                f"Seed Audio returned a non-JSON response (HTTP {response.status_code})."
            ) from error

        if not response.ok:
            message = response_data.get("message", response.text)
            logid = response.headers.get("X-Tt-Logid")
            detail = f" (logid: {logid})" if logid else ""
            raise RuntimeError(
                "Seed Audio request failed (HTTP {}): {}{}".format(
                    response.status_code, message, detail
                )
            )

        code = response_data.get("code")
        if code not in (None, 0):
            raise RuntimeError(
                f"Seed Audio request failed (code={code}): "
                f"{response_data.get('message', 'unknown error')}"
            )

        return self._write_audio_response(response_data, self._output_suffix(request))
