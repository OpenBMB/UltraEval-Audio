"""DashScope API wrapper for qwen-audio-3.1-tts-next.

Despite the shared endpoint path, this is an *audio generation* ("Audiogen")
model rather than a plain TTS model like Qwen-Audio-3.0-TTS
(``audio_evals/models/TTS/qwen_audio_tts.py``). Three things differ:

* the request carries ``input.text_prompt`` -- a free-form scene description --
  instead of ``input.text`` plus a voice ID;
* reference audio is inlined per request as a public URL or a base64 data URI
  and referred to from the prompt as ``@voice1``/``@voice2``/``@voice3``. System
  voices and enrolled voice-clone IDs are not accepted at all, so there is no
  voice-enrollment round trip to make (or clean up);
* only Chinese and English are supported, and there is no ``language_hints``.

Official documentation:
https://help.aliyun.com/zh/model-studio/qwen-audio-3-1-tts-next
https://help.aliyun.com/zh/model-studio/audio-generation
https://help.aliyun.com/zh/model-studio/audio-generation-api
"""

import base64
import logging
import os
import struct
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from audio_evals.base import PromptStruct
from audio_evals.models.model import APIModel

# Shared with the 3.0 wrapper: same vendor, same error envelope, same workspace
# URL shape.
from audio_evals.models.TTS.qwen_audio_tts import (
    WORKSPACE_HTTP_BASE_URL,
    _http_error_message,
)

logger = logging.getLogger(__name__)

SUPPORTED_MODELS = {
    "qwen-audio-3.1-tts-next",
}

# Only the workspace-scoped host is documented for audio generation, so unlike
# the 3.0 wrapper there is no usable public-host default.
SUPPORTED_AUDIO_FORMATS = {"wav", "mp3", "pcm"}
SUPPORTED_SAMPLE_RATES = {8000, 16000, 24000, 44100, 48000}

MAX_TEXT_PROMPT_CHARS = 3000
MAX_REFERENCES = 3
MAX_REFERENCE_BYTES = 10 * 1024 * 1024

# Bare PCM is explicitly rejected by the reference-audio endpoint.
REFERENCE_MIME_TYPES = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".ogg": "audio/ogg",
    ".opus": "audio/ogg",
}

# Used when a prompt supplies plain ``text`` instead of a ready ``text_prompt``.
DEFAULT_CLONE_TEXT_PROMPT = (
    "@voice1 用与参考音频完全相同的音色、口音和语气，清晰、自然地读出："
    "“{text}”。不要添加任何额外内容、背景音或音效。"
)
DEFAULT_TEXT_PROMPT = (
    "一位说话人清晰、自然地读出：“{text}”。" "不要添加任何额外内容、背景音或音效。"
)


class QwenAudioTTSNext(APIModel):
    """qwen-audio-3.1-tts-next API model.

    Prompt fields:
        text_prompt: Full scene description sent as-is. Reference audio is
            addressed positionally as ``@voice1``..``@voice3``.
        instruction: Optional natural-language speaking instruction. The model
            has no separate instruction field (unlike Qwen-Audio-3.0-TTS), so it
            is folded into ``text_prompt``.
        text: Sentence to read, used only when ``text_prompt`` is absent. It is
            wrapped with ``text_prompt_template``.
        prompt_audio: Reference audio -- a local path, a public HTTP(S) URL, or
            a list of up to three of either. Local files are base64-inlined.
        references: Raw ``input.references`` entries, for callers that need to
            bypass ``prompt_audio`` handling.
    """

    def __init__(
        self,
        model_name: str = "qwen-audio-3.1-tts-next",
        workspace_id: Optional[str] = None,
        base_http_url: Optional[str] = None,
        audio_format: str = "wav",
        sample_rate: int = 24000,
        # The API defaults to 2, but every downstream TTS evaluator wants mono.
        channels: int = 1,
        request_timeout: float = 300,
        text_prompt_template: Optional[str] = None,
        sample_params: Optional[Dict[str, Any]] = None,
    ):
        if model_name not in SUPPORTED_MODELS:
            raise ValueError(
                "Unsupported Qwen-Audio-TTS-Next model {!r}; expected one of {}".format(
                    model_name, sorted(SUPPORTED_MODELS)
                )
            )
        if not os.getenv("DASHSCOPE_API_KEY"):
            raise ValueError("DASHSCOPE_API_KEY is required")
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
        if channels not in (1, 2):
            raise ValueError(
                "Unsupported channels {!r}; expected 1 or 2".format(channels)
            )

        workspace_id = workspace_id or os.getenv("DASHSCOPE_WORKSPACE_ID")
        base_http_url = base_http_url or os.getenv("DASHSCOPE_BASE_HTTP_URL")
        if not base_http_url:
            if not workspace_id:
                raise ValueError(
                    "qwen-audio-3.1-tts-next is only documented on the "
                    "workspace-scoped host; set DASHSCOPE_WORKSPACE_ID (or pass "
                    "workspace_id / base_http_url)"
                )
            base_http_url = WORKSPACE_HTTP_BASE_URL.format(workspace_id=workspace_id)

        self.model_name = model_name
        self.base_http_url = base_http_url.rstrip("/")
        self.synthesis_url = (
            f"{self.base_http_url}/services/audio/tts/SpeechSynthesizer"
        )
        self.audio_format = audio_format
        self.sample_rate = sample_rate
        self.channels = channels
        self.request_timeout = request_timeout
        self.text_prompt_template = text_prompt_template
        self.output_suffix = f".{audio_format}"
        super().__init__(is_chat=True, sample_params=sample_params)

    @staticmethod
    def _compose_text_prompt(
        text: str, instruction: Optional[str], has_reference_audio: bool
    ) -> str:
        """Fold an optional speaking instruction into the single free-form prompt.

        Mirrors ``audio_evals.models.TTS.seed_audio.SeedAudio._compose_text_prompt``
        -- the same vendor-agnostic problem, so the same shape. Two differences:
        reference clips are addressed as ``@voice1`` rather than ``@音频1``, and
        the read-aloud text keeps the curly quotes used everywhere else in this
        module (ASCII quotes are ambiguous when the text itself contains one).
        """
        instruction = (instruction or "").strip()
        if instruction and has_reference_audio:
            return (
                "参考@voice1的音色、口音和说话风格。{instruction}\n"
                "朗读内容：“{text}”".format(instruction=instruction, text=text)
            )
        if instruction:
            return "{instruction}\n朗读内容：“{text}”".format(
                instruction=instruction, text=text
            )
        if has_reference_audio:
            return DEFAULT_CLONE_TEXT_PROMPT.format(text=text)
        return DEFAULT_TEXT_PROMPT.format(text=text)

    def _build_reference(self, reference: Any) -> Dict[str, str]:
        if isinstance(reference, dict):
            if not reference.get("audio_url") and not reference.get("audio_data"):
                raise ValueError(
                    "reference entry must set either audio_url or audio_data"
                )
            return dict(reference)
        if not isinstance(reference, str):
            raise TypeError(
                "prompt_audio entries must be a local path, an HTTP(S) URL, or a "
                "reference dict"
            )
        if reference.startswith(("http://", "https://")):
            return {"audio_url": reference}

        path = Path(reference).expanduser()
        if not path.is_file():
            raise FileNotFoundError(
                "Qwen-Audio-TTS-Next reference audio not found: {!r}".format(reference)
            )
        mime_type = REFERENCE_MIME_TYPES.get(path.suffix.lower())
        if mime_type is None:
            raise ValueError(
                "Unsupported reference audio {!r}; expected one of {} (bare PCM is "
                "not accepted)".format(reference, sorted(REFERENCE_MIME_TYPES))
            )
        raw = path.read_bytes()
        if not raw:
            raise ValueError(
                "Qwen-Audio-TTS-Next reference audio is empty: {!r}".format(reference)
            )
        if len(raw) > MAX_REFERENCE_BYTES:
            raise ValueError(
                "Reference audio {!r} is {} bytes, over the {} byte limit".format(
                    reference, len(raw), MAX_REFERENCE_BYTES
                )
            )
        encoded = base64.b64encode(raw).decode("ascii")
        return {"audio_data": f"data:{mime_type};base64,{encoded}"}

    def _build_references(self, prompt_audio: Any, references: Any) -> List[Dict]:
        entries: List[Any] = []
        if references is not None:
            if not isinstance(references, (list, tuple)):
                raise TypeError("references must be a list")
            entries.extend(references)
        if prompt_audio is not None:
            if isinstance(prompt_audio, (list, tuple)):
                entries.extend(prompt_audio)
            else:
                entries.append(prompt_audio)
        # Prompt templates render a missing reference column as "".
        entries = [item for item in entries if item != ""]
        if len(entries) > MAX_REFERENCES:
            raise ValueError(
                "qwen-audio-3.1-tts-next accepts at most {} reference clips, got "
                "{}".format(MAX_REFERENCES, len(entries))
            )
        return [self._build_reference(item) for item in entries]

    @staticmethod
    def _fix_wav_header(audio: bytes) -> bytes:
        """Replace the placeholder chunk sizes in DashScope's streaming WAV.

        The endpoint streams WAV with the RIFF and ``data`` sizes both pinned
        near 2**31-1 rather than the real payload length. libsndfile and ffmpeg
        clamp to the file length, but stdlib ``wave`` trusts the header -- a
        3.2s clip reports as 12.4 hours -- which breaks any consumer deriving a
        duration or buffer size from it.
        """
        if len(audio) < 12 or audio[:4] != b"RIFF" or audio[8:12] != b"WAVE":
            return audio
        patched = bytearray(audio)
        struct.pack_into("<I", patched, 4, len(patched) - 8)
        pos = 12
        while pos + 8 <= len(patched):
            chunk_id = bytes(patched[pos : pos + 4])
            chunk_size = struct.unpack_from("<I", patched, pos + 4)[0]
            if chunk_id == b"data":
                actual = len(patched) - pos - 8
                if chunk_size != actual:
                    logger.info(
                        "Repaired streaming WAV header: data chunk declared %d "
                        "bytes, actual %d",
                        chunk_size,
                        actual,
                    )
                    struct.pack_into("<I", patched, pos + 4, actual)
                break
            pos += 8 + chunk_size + (chunk_size & 1)
        return bytes(patched)

    def _save_audio(self, audio: bytes) -> str:
        if self.audio_format == "wav":
            audio = self._fix_wav_header(audio)
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
            raise TypeError(
                "QwenAudioTTSNext expects a text string or prompt dictionary"
            )
        payload.update(kwargs)

        text_prompt = payload.pop("text_prompt", None)
        text = payload.pop("text", None)
        # Popped before the option whitelist below, since the model has no
        # ``input.instruction`` field to forward it to.
        instruction = payload.pop("instruction", None)
        # Accepted for schema parity with the other voice-clone prompts; the
        # model has no use for the reference transcript.
        payload.pop("prompt_text", None)
        references = self._build_references(
            payload.pop("prompt_audio", None), payload.pop("references", None)
        )

        if isinstance(text_prompt, str) and text_prompt.strip():
            if instruction:
                logger.warning(
                    "Both 'text_prompt' and 'instruction' were supplied; the "
                    "explicit text_prompt wins and the instruction is dropped"
                )
        else:
            if not isinstance(text, str) or not text.strip():
                raise ValueError(
                    "prompt must contain a non-empty 'text_prompt' or 'text' field"
                )
            if self.text_prompt_template:
                text_prompt = self.text_prompt_template.format(text=text)
            else:
                text_prompt = self._compose_text_prompt(
                    text, instruction, bool(references)
                )
        if len(text_prompt) > MAX_TEXT_PROMPT_CHARS:
            raise ValueError(
                "text_prompt is {} characters, over the {} character limit".format(
                    len(text_prompt), MAX_TEXT_PROMPT_CHARS
                )
            )

        allowed_synthesis_options = {
            "volume",
            "rate",
            "speech_rate",
            "seed",
            "enable_cbr",
            "bit_rate",
            "quality",
            "enable_aigc_tag",
            "channels",
            "sample_rate",
            "additional_params",
        }
        unknown_options = set(payload) - allowed_synthesis_options
        if unknown_options:
            raise ValueError(
                "Unsupported Qwen-Audio-TTS-Next options: {}".format(
                    ", ".join(sorted(unknown_options))
                )
            )
        synthesis_options = dict(payload)
        additional_params = synthesis_options.pop("additional_params", None)
        if additional_params:
            if not isinstance(additional_params, dict):
                raise TypeError("additional_params must be a dictionary")
            synthesis_options.update(additional_params)
        # ``speech_rate`` is the 3.0 wrapper's name for the same knob.
        if "speech_rate" in synthesis_options:
            synthesis_options["rate"] = synthesis_options.pop("speech_rate")

        request_input = {
            "text_prompt": text_prompt,
            "format": self.audio_format,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            **synthesis_options,
        }
        if references:
            request_input["references"] = references

        response = requests.post(
            self.synthesis_url,
            headers={
                "Authorization": "Bearer {}".format(os.environ["DASHSCOPE_API_KEY"]),
                "Content-Type": "application/json",
            },
            json={"model": self.model_name, "input": request_input},
            timeout=self.request_timeout,
        )
        if not response.ok:
            summary = {
                "model": self.model_name,
                "format": request_input.get("format"),
                "sample_rate": request_input.get("sample_rate"),
                "channels": request_input.get("channels"),
                "text_prompt_len": len(text_prompt),
                "text_prompt_preview": text_prompt[:200],
                "reference_count": len(references),
            }
            message = _http_error_message(
                response,
                action="SpeechSynthesizer",
                request_summary=summary,
            )
            logger.error(message)
            raise RuntimeError(message)

        result = response.json()
        finish_reason = (result.get("output") or {}).get("finish_reason")
        if finish_reason not in (None, "stop"):
            raise RuntimeError(
                "Qwen-Audio-TTS-Next finished with reason {!r}: {!r}".format(
                    finish_reason, result
                )
            )
        try:
            audio_field = result["output"]["audio"]
        except (KeyError, TypeError) as exc:
            raise RuntimeError(
                "DashScope HTTP response did not contain audio output: {!r}".format(
                    result
                )
            ) from exc

        # The HTTP path returns a 24h URL and leaves ``data`` empty, but accept
        # inline base64 too so the wrapper survives a response-shape switch.
        inline = audio_field.get("data")
        if inline:
            return self._save_audio(base64.b64decode(inline))

        audio_url = audio_field.get("url")
        if not audio_url:
            raise RuntimeError(
                "DashScope HTTP response returned neither audio data nor a URL: "
                "{!r}".format(result)
            )
        audio_response = requests.get(audio_url, timeout=self.request_timeout)
        if not audio_response.ok:
            message = _http_error_message(audio_response, action="audio download")
            logger.error(message)
            raise RuntimeError(message)
        if not audio_response.content:
            raise RuntimeError("Downloaded Qwen-Audio-TTS-Next audio is empty")
        return self._save_audio(audio_response.content)
