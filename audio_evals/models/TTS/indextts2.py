import json
import logging
import select
import uuid
import time
from typing import Dict, Any

from audio_evals.base import PromptStruct
from audio_evals.models.model import OfflineModel
from audio_evals.isolate import isolated

logger = logging.getLogger(__name__)


@isolated("audio_evals/lib/index-tts2/main.py")
class IndexTTS2(OfflineModel):
    """
    Client for interacting with the isolated IndexTTS2 processing script.
    """

    def __init__(
        self,
        model_dir: str,
        config_path: str = "",
        use_fp16: bool = False,
        use_cuda_kernel: bool = False,
        use_deepspeed: bool = False,
        sample_params: Dict[str, Any] = None,
        **kwargs,
    ):
        """
        Initializes the IndexTTS2 client.

        Args:
            model_dir: Path to model checkpoints directory
            config_path: Path to config.yaml file
            use_fp16: Whether to use fp16 inference
            use_cuda_kernel: Whether to use CUDA kernel
            use_deepspeed: Whether to use deepspeed
            sample_params: Sampling parameters
        """
        self.command_args = {
            "model_dir": model_dir,
        }
        if config_path:
            self.command_args["config_path"] = config_path
        if use_fp16:
            self.command_args["use_fp16"] = ""
        if use_cuda_kernel:
            self.command_args["use_cuda_kernel"] = ""
        if use_deepspeed:
            self.command_args["use_deepspeed"] = ""

        super().__init__(is_chat=True, sample_params=sample_params)

    def _inference(self, prompt: PromptStruct, **kwargs) -> str:
        """
        Sends text and audio prompt to the TTS2 server script
        and returns the raw JSON response string containing the output file path.

        Args:
            prompt (PromptStruct): A dictionary containing:
                - 'text' (str): The text to synthesize
                - 'prompt_audio' (str): Path to reference audio file
            **kwargs: Additional keyword arguments

        Returns:
            str: The raw JSON response string from the server script

        Raises:
            RuntimeError: If the TTS script returns an error or times out
            TypeError: If required inputs are not strings
        """
        text = prompt.get("text")
        prompt_audio = prompt.get(
            "prompt_audio", "audio_evals/lib/index-tts2/examples/voice_01.wav"
        )

        if not isinstance(text, str) or not isinstance(prompt_audio, str):
            raise TypeError(
                "Expected 'text' and 'prompt_audio' in prompt to be strings, but got: {}, {}".format(
                    text, prompt_audio
                )
            )

        uid = str(uuid.uuid4())
        prefix = f"{uid}->"

        # Construct JSON request payload
        request_data = {"text": text, "prompt_audio": prompt_audio}
        request_json = json.dumps(request_data)
        request = f"{prefix}{request_json}\n"

        logger.debug(f"Sending request to TTS2 process: {request.strip()}")

        # Send request
        try:
            _, wlist, xlist = select.select(
                [], [self.process.stdin], [self.process.stdin], 60
            )
            if xlist:
                raise RuntimeError("TTS2 stdin broken (select reported error)")
            if not wlist:
                raise TimeoutError("Timeout waiting for TTS2 stdin")
            self.process.stdin.write(request)
            self.process.stdin.flush()
        except BrokenPipeError:
            raise RuntimeError("TTS2 process stdin pipe is broken")
        except Exception as e:
            raise RuntimeError(f"Error writing to TTS2 process stdin: {e}")

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
                        "TTS2 stdout/stderr broken (select reported error)"
                    )

                for read_stream in reads:
                    if read_stream is self.process.stderr:
                        error_output = self.process.stderr.readline().strip()
                        if error_output:
                            logger.error(f"TTS2 stderr: {error_output}")
                    elif read_stream is self.process.stdout:
                        result = self.process.stdout.readline().strip()
                        logger.debug(f"Received from TTS2 process: {result}")
                        if result:
                            if result.startswith(prefix):
                                response_line = result[len(prefix) :]
                                self.process.stdin.write(f"{prefix}ok\n")
                                self.process.stdin.flush()
                                return response_line
                            elif result.startswith("Error:"):
                                raise RuntimeError(f"TTS2 failed: {result}")
                            else:
                                logger.info(result)
            except Exception as e:
                raise RuntimeError(f"Error reading from TTS2 process: {e}")

        if not response_line:
            raise TimeoutError(
                f"Timeout waiting for response from TTS2 process for request {uid}"
            )
