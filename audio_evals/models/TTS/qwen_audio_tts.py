"""DashScope API wrapper for Qwen-Audio-3.0-TTS.

The hosted Plus and Flash models share the same API and support system voices,
instruction control, and voice cloning. Voice enrollment currently requires a
publicly accessible reference-audio URL.

Official documentation:
https://help.aliyun.com/zh/model-studio/cosyvoice-tts-http-api
https://help.aliyun.com/zh/model-studio/voice-cloning-user-guide
"""

import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import requests

from audio_evals.base import PromptStruct
from audio_evals.models.model import APIModel

logger = logging.getLogger(__name__)

SUPPORTED_MODELS = {
    "qwen-audio-3.0-tts-plus",
    "qwen-audio-3.0-tts-flash",
}

DEFAULT_VOICES = {
    "qwen-audio-3.0-tts-plus": "longanlingxin",
    "qwen-audio-3.0-tts-flash": "longanhuan_v3.6",
}

DEFAULT_HTTP_BASE_URL = "https://dashscope.aliyuncs.com/api/v1"
WORKSPACE_HTTP_BASE_URL = "https://{workspace_id}.cn-beijing.maas.aliyuncs.com/api/v1"
SUPPORTED_AUDIO_FORMATS = {"mp3", "pcm", "wav", "opus"}
SUPPORTED_SAMPLE_RATES = {8000, 16000, 22050, 24000, 44100, 48000}


def _http_error_message(
    response: requests.Response,
    *,
    action: str,
    request_summary: Optional[Dict[str, Any]] = None,
    body_limit: int = 4000,
) -> str:
    body = (response.text or "").strip()
    if len(body) > body_limit:
        body = body[:body_limit] + "...(truncated)"
    parts = [
        "Qwen-Audio-TTS {} failed: status={} url={}".format(
            action, response.status_code, response.url
        )
    ]
    if request_summary:
        parts.append("request={!r}".format(request_summary))
    parts.append("body={!r}".format(body))
    return "; ".join(parts)


class QwenAudioTTS(APIModel):
    """Qwen-Audio-3.0-TTS Plus/Flash API model.

    Prompt fields:
        text: Text to synthesize.
        instruction: Optional natural-language speaking instruction.
        voice: Optional system or previously enrolled voice ID.
        prompt_audio: Optional public HTTP(S) reference-audio URL. When set, a
            temporary cloned voice is enrolled and used for this request.

    For a local ``prompt_audio`` file, the wrapper uploads it through the
    DashScope Files HTTP API and passes the resulting signed URL to voice
    enrollment. Temporary uploaded files and cloned voices are deleted after
    synthesis.
    """

    def __init__(
        self,
        model_name: str,
        voice: Optional[str] = None,
        workspace_id: Optional[str] = None,
        base_http_url: Optional[str] = None,
        audio_format: str = "wav",
        sample_rate: int = 24000,
        request_timeout: float = 180,
        delete_cloned_voice: bool = True,
        clone_prefix: str = "ultraeval",
        sample_params: Optional[Dict[str, Any]] = None,
    ):
        if model_name not in SUPPORTED_MODELS:
            raise ValueError(
                "Unsupported Qwen-Audio-TTS model {!r}; expected one of {}".format(
                    model_name, sorted(SUPPORTED_MODELS)
                )
            )
        if not os.getenv("DASHSCOPE_API_KEY"):
            raise ValueError("DASHSCOPE_API_KEY is required")
        if not clone_prefix.isalnum() or len(clone_prefix) > 10:
            raise ValueError(
                "clone_prefix must be alphanumeric and at most 10 characters"
            )
        if audio_format not in SUPPORTED_AUDIO_FORMATS:
            raise ValueError(
                "Unsupported audio_format {!r}; expected one of {}".format(
                    audio_format, sorted(SUPPORTED_AUDIO_FORMATS)
                )
            )
        if sample_rate not in SUPPORTED_SAMPLE_RATES:
            raise ValueError(
                "Unsupported sample_rate {!r}; expected one of {}".format(
                    sample_rate, sorted(SUPPORTED_SAMPLE_RATES)
                )
            )

        workspace_id = workspace_id or os.getenv("DASHSCOPE_WORKSPACE_ID")
        if workspace_id:
            default_http_url = WORKSPACE_HTTP_BASE_URL.format(workspace_id=workspace_id)
        else:
            default_http_url = DEFAULT_HTTP_BASE_URL

        self.model_name = model_name
        self.voice = voice or DEFAULT_VOICES[model_name]
        self.base_http_url = (
            base_http_url or os.getenv("DASHSCOPE_BASE_HTTP_URL") or default_http_url
        ).rstrip("/")
        self.synthesis_url = (
            f"{self.base_http_url}/services/audio/tts/SpeechSynthesizer"
        )
        self.delete_cloned_voice = delete_cloned_voice
        self.clone_prefix = clone_prefix
        self.audio_format = audio_format
        self.sample_rate = sample_rate
        self.request_timeout = request_timeout
        self.output_suffix = ".ogg" if audio_format == "opus" else f".{audio_format}"
        super().__init__(is_chat=True, sample_params=sample_params)

    def _configure_dashscope_http(self):
        import dashscope

        dashscope.api_key = os.environ["DASHSCOPE_API_KEY"]
        dashscope.base_http_api_url = self.base_http_url
        return dashscope

    def _prepare_reference_audio(self, reference_audio: str):
        if not isinstance(reference_audio, str):
            raise TypeError("prompt_audio must be a local path or HTTP(S) URL string")
        if reference_audio.startswith(("http://", "https://")):
            return reference_audio, None

        path = Path(reference_audio).expanduser()
        if not path.is_file():
            raise FileNotFoundError(
                "Qwen-Audio-TTS reference audio not found: {!r}".format(reference_audio)
            )

        self._configure_dashscope_http()
        from dashscope import Files

        upload = Files.upload(str(path), purpose="inference")
        if upload.status_code != 200:
            raise RuntimeError(
                "Failed to upload Qwen-Audio-TTS reference audio: {} {}".format(
                    upload.code, upload.message
                )
            )
        try:
            file_id = upload.output["uploaded_files"][0]["file_id"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                "DashScope file upload response did not contain a file ID"
            ) from exc

        try:
            file_info = Files.get(file_id)
            if file_info.status_code != 200:
                raise RuntimeError(
                    "Failed to query uploaded Qwen-Audio-TTS reference audio: "
                    "{} {}".format(file_info.code, file_info.message)
                )
            reference_url = file_info.output["url"]
            if not reference_url:
                raise RuntimeError(
                    "DashScope uploaded reference audio returned an empty URL"
                )
            logger.info(
                "Uploaded local Qwen-Audio-TTS reference audio as file %s", file_id
            )
            return reference_url, file_id
        except Exception:
            Files.delete(file_id)
            raise

    def _create_cloned_voice(
        self,
        reference_audio: str,
        language_hints: Optional[list] = None,
        max_prompt_audio_length: Optional[float] = None,
        enable_preprocess: Optional[bool] = None,
    ):
        reference_url, uploaded_file_id = self._prepare_reference_audio(reference_audio)
        self._configure_dashscope_http()
        from dashscope import Files
        from dashscope.audio.tts_v2 import VoiceEnrollmentService

        service = VoiceEnrollmentService()
        create_kwargs = {
            "target_model": self.model_name,
            "prefix": self.clone_prefix,
            "url": reference_url,
        }
        if language_hints is not None:
            create_kwargs["language_hints"] = language_hints
        if max_prompt_audio_length is not None:
            create_kwargs["max_prompt_audio_length"] = max_prompt_audio_length
        if enable_preprocess is not None:
            create_kwargs["enable_preprocess"] = enable_preprocess
        try:
            voice_id = service.create_voice(**create_kwargs)
        except Exception:
            if uploaded_file_id is not None:
                Files.delete(uploaded_file_id)
            raise
        logger.info("Created temporary Qwen-Audio-TTS voice %s", voice_id)
        return service, voice_id, uploaded_file_id

    def _save_audio(self, audio: bytes) -> str:
        with tempfile.NamedTemporaryFile(
            suffix=self.output_suffix, delete=False
        ) as output:
            output.write(audio)
            return output.name

    def _inference(self, prompt: PromptStruct, **kwargs) -> str:
        if isinstance(prompt, str):
            payload = {"text": prompt}
        elif isinstance(prompt, dict):
            payload = dict(prompt)
        else:
            raise TypeError("QwenAudioTTS expects a text string or prompt dictionary")
        payload.update(kwargs)

        text = payload.pop("text", None)
        if not isinstance(text, str) or not text.strip():
            raise ValueError("prompt must contain a non-empty string field 'text'")

        instruction = payload.pop("instruction", None)
        reference_audio = payload.pop(
            "prompt_audio", payload.pop("reference_audio_url", None)
        )
        voice = payload.pop("voice", self.voice)
        payload.pop("prompt_text", None)

        language_hints = payload.pop("language_hints", None)
        clone_options = {
            "language_hints": language_hints,
            "max_prompt_audio_length": payload.pop("max_prompt_audio_length", None),
            "enable_preprocess": payload.pop("enable_preprocess", None),
        }
        allowed_synthesis_options = {
            "volume",
            "speech_rate",
            "pitch_rate",
            "rate",
            "pitch",
            "bit_rate",
            "enable_ssml",
            "seed",
            "enable_aigc_tag",
            "aigc_propagator",
            "aigc_propagate_id",
            "additional_params",
        }
        unknown_options = set(payload) - allowed_synthesis_options
        if unknown_options:
            raise ValueError(
                "Unsupported Qwen-Audio-TTS options: {}".format(
                    ", ".join(sorted(unknown_options))
                )
            )
        synthesis_options = {
            key: value
            for key, value in payload.items()
            if key in allowed_synthesis_options
        }
        additional_params = synthesis_options.pop("additional_params", None)
        if additional_params:
            if not isinstance(additional_params, dict):
                raise TypeError("additional_params must be a dictionary")
            synthesis_options.update(additional_params)
        if "speech_rate" in synthesis_options:
            synthesis_options["rate"] = synthesis_options.pop("speech_rate")
        if "pitch_rate" in synthesis_options:
            synthesis_options["pitch"] = synthesis_options.pop("pitch_rate")
        if language_hints is not None:
            synthesis_options["language_hints"] = language_hints
        if instruction is not None:
            synthesis_options["instruction"] = instruction

        enrollment_service = None
        cloned_voice_id = None
        uploaded_file_id = None
        try:
            if reference_audio:
                (
                    enrollment_service,
                    cloned_voice_id,
                    uploaded_file_id,
                ) = self._create_cloned_voice(reference_audio, **clone_options)
                voice = cloned_voice_id
            request_input = {
                "text": text,
                "voice": voice,
                "format": self.audio_format,
                "sample_rate": self.sample_rate,
                **synthesis_options,
            }
            request_payload = {"model": self.model_name, "input": request_input}
            response = requests.post(
                self.synthesis_url,
                headers={
                    "Authorization": "Bearer {}".format(
                        os.environ["DASHSCOPE_API_KEY"]
                    ),
                    "Content-Type": "application/json",
                },
                json=request_payload,
                timeout=self.request_timeout,
            )
            if not response.ok:
                summary = {
                    "model": self.model_name,
                    "voice": request_input.get("voice"),
                    "format": request_input.get("format"),
                    "sample_rate": request_input.get("sample_rate"),
                    "text_len": len(text),
                    "text_preview": text[:200],
                    "has_instruction": "instruction" in request_input,
                    "used_cloned_voice": bool(reference_audio),
                }
                message = _http_error_message(
                    response,
                    action="SpeechSynthesizer",
                    request_summary=summary,
                )
                logger.error(message)
                raise RuntimeError(message)
            result = response.json()
            try:
                audio_url = result["output"]["audio"]["url"]
            except (KeyError, TypeError) as exc:
                raise RuntimeError(
                    "DashScope HTTP response did not contain an audio URL: {!r}".format(
                        result
                    )
                ) from exc
            if not audio_url:
                raise RuntimeError(
                    "DashScope HTTP response returned an empty audio URL"
                )

            audio_response = requests.get(
                audio_url,
                timeout=self.request_timeout,
            )
            if not audio_response.ok:
                message = _http_error_message(
                    audio_response,
                    action="audio download",
                )
                logger.error(message)
                raise RuntimeError(message)
            audio = audio_response.content
            if not audio:
                raise RuntimeError("Downloaded Qwen-Audio-TTS audio is empty")
            return self._save_audio(audio)
        finally:
            if (
                enrollment_service is not None
                and cloned_voice_id is not None
                and self.delete_cloned_voice
            ):
                try:
                    enrollment_service.delete_voice(cloned_voice_id)
                    logger.info(
                        "Deleted temporary Qwen-Audio-TTS voice %s", cloned_voice_id
                    )
                except Exception:
                    logger.warning(
                        "Failed to delete temporary Qwen-Audio-TTS voice %s",
                        cloned_voice_id,
                        exc_info=True,
                    )
            if uploaded_file_id is not None:
                try:
                    from dashscope import Files

                    Files.delete(uploaded_file_id)
                    logger.info(
                        "Deleted temporary DashScope reference file %s",
                        uploaded_file_id,
                    )
                except Exception:
                    logger.warning(
                        "Failed to delete temporary DashScope reference file %s",
                        uploaded_file_id,
                        exc_info=True,
                    )
