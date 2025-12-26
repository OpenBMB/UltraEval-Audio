import json
import logging
import select
import uuid
import time
from typing import Dict, Any

from audio_evals.base import PromptStruct
from audio_evals.models.model import OfflineModel
from audio_evals.isolate import isolated
import os

logger = logging.getLogger(__name__)


@isolated(
    "audio_evals/lib/GLM-TTS/main.py",
    pre_command="mkdir -p ./third_party && ([ ! -d './third_party/GLM-TTS' ] && "
    "git clone https://github.com/UltraEval/GLM-TTS.git ./third_party/GLM-TTS) || true && pwd",
)
class GLMTTS(OfflineModel):
    """
    Client for interacting with the isolated GLM-TTS processing script.

    GLM-TTS is a high-quality text-to-speech synthesis system based on large
    language models, supporting zero-shot voice cloning and streaming inference.
    """

    def __init__(
        self,
        ckpt_dir: str,
        use_cache: bool = True,
        use_phoneme: bool = False,
        sample_params: Dict[str, Any] = None,
        **kwargs,
    ):
        """
        Initialize the GLM-TTS client.

        Args:
            ckpt_dir: Path to GLM-TTS checkpoint directory
            use_cache: Whether to use KV cache for faster inference
            use_phoneme: Whether to enable phoneme input mode for precise pronunciation
            sample_params: Additional sampling parameters
        """
        # ckpt_dir is relative to third_party/GLM-TTS/ (e.g., "ckpt")
        # Model weights should be downloaded to third_party/GLM-TTS/ckpt/
        if not os.path.exists(ckpt_dir):
            ckpt_dir = self._download_model(ckpt_dir)
        self.command_args = {
            "ckpt_dir": ckpt_dir,
        }

        if use_cache:
            self.command_args["use_cache"] = ""
        if use_phoneme:
            self.command_args["use_phoneme"] = ""

        super().__init__(is_chat=True, sample_params=sample_params)

    def _inference(self, prompt: PromptStruct, **kwargs) -> str:
        """
        Send text and audio prompt to the GLM-TTS server script
        and return the raw JSON response string containing the output file path.

        Args:
            prompt (PromptStruct): A dictionary containing:
                - 'text' (str): The text to synthesize
                - 'prompt_audio' (str): Path to reference audio file
                - 'prompt_text' (str, optional): Transcript of reference audio
            **kwargs: Additional keyword arguments

        Returns:
            str: The raw JSON response string from the server script

        Raises:
            RuntimeError: If the TTS script returns an error or times out
            TypeError: If required inputs are not strings
        """
        text = prompt.get("text")
        prompt_audio = prompt.get("prompt_audio")
        prompt_text = prompt.get("prompt_text", None)

        if not isinstance(text, str):
            raise TypeError(
                f"Expected 'text' in prompt to be string, but got: {type(text)}"
            )
        if not isinstance(prompt_audio, str):
            raise TypeError(
                f"Expected 'prompt_audio' in prompt to be string, but got: {type(prompt_audio)}"
            )

        uid = str(uuid.uuid4())
        prefix = f"{uid}->"

        # Construct JSON request payload
        request_data = {
            "text": text,
            "prompt_audio": prompt_audio,
        }
        if prompt_text:
            request_data["prompt_text"] = prompt_text

        request_json = json.dumps(request_data, ensure_ascii=False)
        request = f"{prefix}{request_json}\n"

        logger.debug(f"Sending request to GLM-TTS process: {request.strip()}")

        # Send request
        try:
            _, wlist, xlist = select.select(
                [], [self.process.stdin], [self.process.stdin], 180
            )
            if xlist:
                raise RuntimeError("GLM-TTS stdin broken (select reported error)")
            if not wlist:
                raise TimeoutError("Timeout waiting for GLM-TTS stdin")
            self.process.stdin.write(request)
            self.process.stdin.flush()
        except BrokenPipeError:
            raise RuntimeError("GLM-TTS process stdin pipe is broken")
        except Exception as e:
            raise RuntimeError(f"Error writing to GLM-TTS process stdin: {e}")

        # Receive response
        max_wait_time = 300  # Longer timeout for TTS generation
        start_time = time.time()
        response_line = None

        while time.time() - start_time < max_wait_time:
            try:
                reads, _, xlist = select.select(
                    [self.process.stdout, self.process.stderr],
                    [],
                    [self.process.stdout, self.process.stderr],
                    1.0,
                )
                if xlist:
                    raise RuntimeError(
                        "GLM-TTS stdout/stderr broken (select reported error)"
                    )

                for read_stream in reads:
                    if read_stream is self.process.stderr:
                        error_output = self.process.stderr.readline().strip()
                        if error_output:
                            logger.error(f"GLM-TTS stderr: {error_output}")
                    elif read_stream is self.process.stdout:
                        result = self.process.stdout.readline().strip()
                        if result:
                            if result.startswith(prefix):
                                response_line = result[len(prefix) :]
                                self.process.stdin.write(f"{prefix}close\n")
                                self.process.stdin.flush()
                                return response_line
                            elif result.startswith("Error:"):
                                raise RuntimeError(f"GLM-TTS failed: {result}")
                            else:
                                logger.info(result)
            except Exception as e:
                raise RuntimeError(f"Error reading from GLM-TTS process: {e}")

        if not response_line:
            raise TimeoutError(
                f"Timeout waiting for response from GLM-TTS process for request {uid}"
            )
