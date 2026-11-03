import logging
import os
import tempfile
from typing import Dict, Optional

from openai import OpenAI

from audio_evals.base import PromptStruct
from audio_evals.models.model import APIModel

logger = logging.getLogger(__name__)


class OpenAITTS(APIModel):
    """OpenAI Text-to-Speech API model (the ``audio/speech`` endpoint).

    Supports ``gpt-4o-mini-tts`` (with ``instructions`` style/emotion control),
    as well as ``tts-1`` / ``tts-1-hd``. Returns the file path of the generated
    audio.

    Reference: https://platform.openai.com/docs/guides/text-to-speech
    """

    def __init__(
        self,
        model_name: str = "gpt-4o-mini-tts",
        voice: str = "coral",
        response_format: str = "wav",
        instructions: Optional[str] = None,
        speed: Optional[float] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        sample_params: Optional[Dict[str, any]] = None,
    ):
        """
        Args:
            model_name: "gpt-4o-mini-tts", "tts-1", or "tts-1-hd"
            voice: built-in voice, e.g. alloy/ash/ballad/coral/echo/fable/onyx/nova/sage/shimmer/verse
            response_format: mp3/opus/aac/flac/wav/pcm
            instructions: style/tone/emotion guidance (gpt-4o-mini-tts only; ignored by tts-1/tts-1-hd)
            speed: 0.25-4.0 (tts-1/tts-1-hd only; gpt-4o-mini-tts controls speed via instructions)
        """
        super().__init__(is_chat=True, sample_params=sample_params)
        self.model_name = model_name
        self.voice = voice
        self.response_format = response_format
        self.instructions = instructions
        self.speed = speed

        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key is required. "
                "Pass api_key or set the OPENAI_API_KEY environment variable."
            )
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    @staticmethod
    def _extract_text(prompt: PromptStruct) -> str:
        if isinstance(prompt, dict):
            return prompt.get("text", "")
        if isinstance(prompt, list):
            for msg in prompt:
                for c in msg.get("contents", []):
                    if c.get("type") == "text":
                        return c.get("value", "")
            return ""
        return str(prompt)

    @staticmethod
    def _extract_instruction(prompt: PromptStruct) -> Optional[str]:
        """Pull a per-sample style/language instruction out of the prompt.

        Uses the ``instruction`` key that the instruct-TTS prompt templates
        already carry (see ``registry/prompt/instruct_tts_eval.yaml`` and
        ``TTS/voxcpm2.py``), so those templates work here unchanged.
        """
        if isinstance(prompt, dict):
            return prompt.get("instruction") or None
        if isinstance(prompt, list):
            for msg in prompt:
                for c in msg.get("contents", []):
                    if c.get("key") == "instruction":
                        return c.get("value") or None
        return None

    def _inference(self, prompt: PromptStruct, **kwargs) -> str:
        text = self._extract_text(prompt)
        if not text:
            raise ValueError("No text found in prompt for TTS")

        voice = kwargs.pop("voice", self.voice)
        response_format = kwargs.pop("response_format", self.response_format)
        # `instructions` is a per-sample signal, so the prompt (i.e. the dataset
        # row / the chosen template) wins over sample_params, which in turn wins
        # over the model-level default. Mirrors the precedence used by mb_tts.
        kw_instructions = kwargs.pop("instructions", None)
        instructions = (
            self._extract_instruction(prompt) or kw_instructions or self.instructions
        )
        speed = kwargs.pop("speed", self.speed)

        request_params = dict(
            model=self.model_name,
            voice=voice,
            input=text,
            response_format=response_format,
        )
        # instructions only supported by gpt-4o-mini-tts (not tts-1/tts-1-hd)
        if instructions:
            request_params["instructions"] = instructions
        # speed not supported by gpt-4o-mini-tts; pass only when explicitly set
        if speed is not None:
            request_params["speed"] = speed
        request_params.update(kwargs)

        suffix = f".{response_format}"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            out_path = f.name

        logger.info(
            "OpenAI TTS request: model=%s, chars=%d, voice=%s, format=%s, instructions=%r",
            self.model_name,
            len(text),
            voice,
            response_format,
            instructions,
        )

        with self.client.audio.speech.with_streaming_response.create(
            **request_params
        ) as response:
            response.stream_to_file(out_path)

        logger.info(f"OpenAI TTS audio saved to {out_path}")
        return out_path
