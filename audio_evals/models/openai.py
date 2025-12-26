import os
import base64
import json
import tempfile
import subprocess
import logging
from typing import Dict, Any, List
from urllib.parse import urlparse
import requests
from openai import OpenAI, AzureOpenAI
from azure.core.credentials import AzureKeyCredential


from audio_evals.models.model import APIModel
from audio_evals.base import PromptStruct

logger = logging.getLogger(__name__)


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
        return transcript.text


class AdvancedGPT(APIModel):
    """
    Advanced GPT model supporting audio input and output via OpenAI Chat Completions API.
    Uses gpt-4o-audio-preview model for multimodal audio capabilities.

    Reference: https://platform.openai.com/docs/guides/audio

    Supported models:
    - gpt-4o-audio-preview: Full audio input/output support
    - gpt-4o-audio-preview-2024-12-17: Latest version with improved voice quality
    - gpt-4o-mini-audio-preview: Smaller, faster model
    - gpt-4o-mini-audio-preview-2024-12-17: Latest mini version
    """

    def __init__(
        self,
        model_name: str,
        modalities: List[str] = None,
        voice: str = "alloy",
        audio_format: str = "wav",
        base_url: str = "https://api.openai.com/v1",
        api_key: str = None,
        sample_params: Dict[str, Any] = None,
    ):
        """
        Initialize AdvancedGPT model.

        Args:
            model_name: Model name (e.g., "gpt-4o-audio-preview")
            modalities: Output modalities, e.g. ["text"], ["text", "audio"]
            voice: Voice for audio output, options: "alloy", "ash", "ballad", "coral", "echo", "sage", "shimmer", "verse"
            audio_format: Audio output format, options: "wav", "mp3", "flac", "opus", "pcm16"
            sample_params: Additional sampling parameters
        """
        super().__init__(True, sample_params)
        self.model_name = model_name
        self.modalities = modalities if modalities else ["text"]
        self.voice = voice
        self.audio_format = audio_format

        if api_key is None:
            api_key = os.environ["OPENAI_API_KEY"]
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    # OpenAI audio API only supports wav and mp3 formats
    SUPPORTED_AUDIO_FORMATS = {".wav", ".mp3"}

    def _is_url(self, path: str) -> bool:
        """Check if the path is a URL."""
        return path.startswith(("http://", "https://"))

    def _get_audio_ext(self, audio_path: str) -> str:
        """Get audio file extension from file path or URL."""
        if self._is_url(audio_path):
            parsed = urlparse(audio_path)
            path = parsed.path
        else:
            path = audio_path
        return os.path.splitext(path)[1].lower()

    def _convert_to_wav(self, audio_data: bytes, original_ext: str) -> bytes:
        """Convert audio data to wav format using ffmpeg."""
        with tempfile.NamedTemporaryFile(
            suffix=original_ext, delete=False
        ) as input_file:
            input_file.write(audio_data)
            input_path = input_file.name

        output_path = input_path.rsplit(".", 1)[0] + ".wav"

        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    input_path,
                    "-acodec",
                    "pcm_s16le",
                    "-ar",
                    "16000",
                    "-ac",
                    "1",
                    output_path,
                ],
                capture_output=True,
                check=True,
            )
            with open(output_path, "rb") as f:
                wav_data = f.read()
            return wav_data
        finally:
            # Clean up temp files
            if os.path.exists(input_path):
                os.remove(input_path)
            if os.path.exists(output_path):
                os.remove(output_path)

    def _load_audio_data(self, audio_path: str) -> bytes:
        """Load audio data from local file or URL."""
        if self._is_url(audio_path):
            response = requests.get(audio_path, timeout=30)
            response.raise_for_status()
            return response.content
        else:
            with open(audio_path, "rb") as audio_file:
                return audio_file.read()

    def _encode_audio(self, audio_path: str) -> tuple:
        """
        Encode audio file to base64 string. Supports both local files and URLs.
        Converts unsupported formats (non wav/mp3) to wav using ffmpeg.

        Returns:
            tuple: (base64_encoded_data, audio_format)
        """
        ext = self._get_audio_ext(audio_path)
        audio_data = self._load_audio_data(audio_path)

        # Convert to wav if format is not supported
        if ext not in self.SUPPORTED_AUDIO_FORMATS:
            logger.info(f"Converting audio from {ext} to wav format")
            audio_data = self._convert_to_wav(audio_data, ext)
            audio_format = "wav"
        else:
            audio_format = "wav" if ext == ".wav" else "mp3"

        return base64.b64encode(audio_data).decode("utf-8"), audio_format

    def _build_message_content(self, contents: List[Dict]) -> List[Dict]:
        """
        Build OpenAI message content from PromptStruct contents.
        Handles text and audio content types.
        """
        message_content = []
        for item in contents:
            content_type = item.get("type")
            value = item.get("value")

            if content_type == "text":
                message_content.append({"type": "text", "text": value})
            elif content_type == "audio":
                audio_data, audio_format = self._encode_audio(value)
                message_content.append(
                    {
                        "type": "input_audio",
                        "input_audio": {"data": audio_data, "format": audio_format},
                    }
                )
            elif content_type == "image":
                # Also support image input for multimodal scenarios
                if value.startswith(("http://", "https://")):
                    message_content.append(
                        {"type": "image_url", "image_url": {"url": value}}
                    )
                else:
                    with open(value, "rb") as img_file:
                        img_data = base64.b64encode(img_file.read()).decode("utf-8")
                    ext = os.path.splitext(value)[1].lower()
                    mime_type = {
                        ".png": "image/png",
                        ".jpg": "image/jpeg",
                        ".jpeg": "image/jpeg",
                        ".gif": "image/gif",
                        ".webp": "image/webp",
                    }.get(ext, "image/png")
                    message_content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime_type};base64,{img_data}"},
                        }
                    )

        return message_content

    def _inference(self, prompt: PromptStruct, **kwargs) -> str:
        """
        Perform inference with audio support.

        Args:
            prompt: PromptStruct containing messages with text/audio content
            **kwargs: Additional arguments passed to API

        Returns:
            str: Text response or JSON with audio path and text if audio output is enabled
        """
        messages = []
        for item in prompt:
            role = item["role"]
            contents = item.get("contents", [])

            # Handle simple text content (backward compatibility)
            if len(contents) == 1 and contents[0].get("type") == "text":
                messages.append({"role": role, "content": contents[0]["value"]})
            else:
                # Handle multimodal content
                message_content = self._build_message_content(contents)
                messages.append({"role": role, "content": message_content})

        # Build request parameters
        request_params = {
            "model": self.model_name,
            "messages": messages,
            "modalities": self.modalities,
        }

        # Add audio configuration if audio output is requested
        if "audio" in self.modalities:
            request_params["audio"] = {"voice": self.voice, "format": self.audio_format}

        # Merge with additional kwargs
        request_params.update(kwargs)

        response = self.client.chat.completions.create(**request_params)

        # Process response
        message = response.choices[0].message
        text_response = message.content

        # Handle audio output
        if "audio" in self.modalities and hasattr(message, "audio") and message.audio:
            audio_data = message.audio.data
            audio_transcript = getattr(message.audio, "transcript", "")

            # Decode and save audio to temp file
            audio_bytes = base64.b64decode(audio_data)
            save_path = os.path.join(os.getcwd(), "tmp/")
            os.makedirs(save_path, exist_ok=True)

            suffix = f".{self.audio_format}" if self.audio_format != "pcm16" else ".pcm"
            with tempfile.NamedTemporaryFile(
                suffix=suffix, delete=False, dir=save_path
            ) as f:
                f.write(audio_bytes)
                audio_path = f.name

            return json.dumps(
                {"audio": audio_path, "text": text_response or audio_transcript},
                ensure_ascii=False,
            )

        return text_response
