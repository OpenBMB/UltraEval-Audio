"""
Qwen3-TTS model wrappers for UltraEval-Audio.

Supports three modes:
- custom_voice: Use preset speakers with optional style instructions
- voice_design: Create voices from natural language descriptions
- voice_clone: Clone voices from reference audio

Reference: https://github.com/QwenLM/Qwen3-TTS
"""

import json
import logging
import os
import select
from typing import Dict, Optional

from audio_evals.base import PromptStruct
from audio_evals.isolate import isolated
from audio_evals.models.model import OfflineModel

logger = logging.getLogger(__name__)


@isolated("audio_evals/lib/Qwen3TTS/main.py", pre_command="uv pip install -U qwen-tts")
class Qwen3TTS(OfflineModel):
    """
    Qwen3-TTS unified model.
    
    Supports three modes via the `mode` parameter:
    - custom_voice: Use preset speakers with optional style control
    - voice_design: Create voices from natural language descriptions
    - voice_clone: Clone voices from reference audio
    
    Available speakers (for custom_voice mode):
    - Vivian: Bright, slightly edgy young female voice (Chinese)
    - Serena: Warm, gentle young female voice (Chinese)
    - Uncle_Fu: Seasoned male voice with a low, mellow timbre (Chinese)
    - Dylan: Youthful Beijing male voice (Chinese, Beijing Dialect)
    - Eric: Lively Chengdu male voice (Chinese, Sichuan Dialect)
    - Ryan: Dynamic male voice with strong rhythmic drive (English)
    - Aiden: Sunny American male voice (English)
    - Ono_Anna: Playful Japanese female voice (Japanese)
    - Sohee: Warm Korean female voice (Korean)
    """

    def __init__(
        self,
        path: str,
        mode: str = "custom_voice",
        dtype: str = "bfloat16",
        device: str = "cuda:0",
        sample_params: Optional[Dict] = None,
        *args,
        **kwargs,
    ):
        """
        Initialize Qwen3-TTS model.
        
        Args:
            path: Model path or HuggingFace model ID
            mode: Generation mode - "custom_voice", "voice_design", or "voice_clone"
            dtype: Model dtype - "float16", "bfloat16", or "float32"
            device: Device to run on, e.g., "cuda:0"
            sample_params: Additional sampling parameters
        """
        if mode not in ("custom_voice", "voice_design", "voice_clone"):
            raise ValueError(f"Invalid mode: {mode}. Must be one of: custom_voice, voice_design, voice_clone")
        
        if not os.path.exists(path):
            path = self._download_model(path)

        self.command_args = {
            "path": path,
            "mode": mode,
            "dtype": dtype,
            "device": device,
        }
        super().__init__(is_chat=True, sample_params=sample_params)

    def _inference(self, prompt: PromptStruct, **kwargs) -> str:
        import uuid

        uid = str(uuid.uuid4())
        prefix = f"{uid}->"
        
        # Merge prompt dict with kwargs
        if isinstance(prompt, dict):
            prompt.update(kwargs)
        else:
            prompt = {"text": prompt, **kwargs}

        while True:
            _, wlist, _ = select.select([], [self.process.stdin], [], 180)
            if not wlist:
                raise RuntimeError("Write timeout after 180 seconds")
            try:
                self.process.stdin.write(
                    f"{prefix}{json.dumps(prompt, ensure_ascii=False)}\n"
                )
                self.process.stdin.flush()
                logger.debug("prompt written to Qwen3-TTS stdin")
                break
            except BlockingIOError:
                continue

        while True:
            rlist, _, _ = select.select(
                [self.process.stdout, self.process.stderr], [], [], 300
            )
            if not rlist:
                err_msg = "Read timeout after 300 seconds"
                logger.error(err_msg)
                raise RuntimeError(err_msg)

            try:
                for stream in rlist:
                    if stream == self.process.stdout:
                        result = self.process.stdout.readline().strip()
                        if not result:
                            continue
                        if result.startswith(prefix):
                            self.process.stdin.write(f"{prefix}close\n")
                            self.process.stdin.flush()
                            return result[len(prefix):]
                        elif result.startswith("Error:"):
                            raise RuntimeError(f"Qwen3-TTS failed: {result}")
                        else:
                            logger.info(result)
                    elif stream == self.process.stderr:
                        err = self.process.stderr.readline().strip()
                        if err:
                            logger.error(f"Process stderr: {err}")
            except BlockingIOError as e:
                logger.error(f"BlockingIOError occurred: {str(e)}")
                continue
