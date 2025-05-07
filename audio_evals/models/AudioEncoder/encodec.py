import json
import logging
import select
import uuid
import time
from typing import Dict, Any

from audio_evals.base import PromptStruct
from audio_evals.models.model import OfflineModel  # Changed from Model to OfflineModel
from audio_evals.isolate import isolated

logger = logging.getLogger(__name__)


@isolated("audio_evals/lib/encodec/main.py")  # Point to the new server script
class Encodec(OfflineModel):  # Inherit from OfflineModel for process handling
    """
    Client for interacting with the isolated Encodec processing script.
    """

    def __init__(
        self,
        path: str,  # Renamed 'path' to 'model_path' for clarity and consistency
        sample_params: Dict[str, any] = None,
        *args,  # Capture potential extra args for the isolated script
        **kwargs,  # Capture potential extra keyword args
    ):
        """
        Initializes the Encodec client.

        Args:
            path (str): Path to the pretrained Encodec model directory or file,
                              passed to the server script.
            sample_params (Dict, optional): Sampling parameters. Defaults to None.
        """
        self.command_args = {
            "path": path,
        }
        # Include any other necessary arguments passed via kwargs for the isolated script
        # (Currently, main.py only takes model_path, but this allows future flexibility)
        # Encodec client returns a string (JSON), not interactive chat
        super().__init__(is_chat=True, sample_params=sample_params)

    def _inference(self, prompt: PromptStruct, **kwargs) -> str:
        """
        Sends an audio file path and processing flags to the Encodec server script
        and returns the raw JSON response string containing the output file path or error.

        Args:
            prompt (PromptStruct): A dictionary containing:
                - 'audio' (str): The input audio file path.
                - Optional: 'mono' (bool): Force mono processing.
                - Optional: 'stereo' (bool): Force stereo processing.
            **kwargs: Additional keyword arguments potentially containing 'mono' or 'stereo'.

        Returns:
            str: The raw JSON response string from the server script.

        Raises:
            RuntimeError: If the Encodec script returns an error or times out.
            TypeError: If the audio path is not a string.
        """
        audio_filepath = prompt.get("audio")
        if not isinstance(audio_filepath, str):
            raise TypeError(
                f"Expected 'audio' in prompt to be a string, got {type(audio_filepath)}"
            )

        # Determine mono/stereo flags from prompt or kwargs (kwargs take precedence)
        mono_flag = "stereo" not in kwargs
        stereo_flag = kwargs.get("stereo", False)
        if mono_flag and stereo_flag:
            logger.warning("Both mono and stereo flags are set; defaulting to mono.")
            stereo_flag = False  # Prioritize mono if both are true

        uid = str(uuid.uuid4())
        prefix = f"{uid}->"

        # Construct JSON request payload
        request_data = {
            "audio": audio_filepath,
            "mono": mono_flag,
            "stereo": stereo_flag,
        }
        request_json = json.dumps(request_data)
        request = f"{prefix}{request_json}\n"

        logger.debug(f"Sending request to Encodec process: {request.strip()}")

        # Send request
        try:
            _, wlist, xlist = select.select(
                [], [self.process.stdin], [self.process.stdin], 60
            )
            if xlist:
                raise RuntimeError("Encodec stdin broken (select reported error).")
            if not wlist:
                raise TimeoutError("Timeout waiting for Encodec stdin.")
            self.process.stdin.write(request)
            self.process.stdin.flush()
        except BrokenPipeError:
            raise RuntimeError("Encodec process stdin pipe is broken.")
        except Exception as e:
            raise RuntimeError(f"Error writing to Encodec process stdin: {e}")

        # Receive response
        max_wait_time = 120  # Adjust timeout as needed for encoding
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
                        "Encodec stdout/stderr broken (select reported error)."
                    )

                for read_stream in reads:
                    if read_stream is self.process.stderr:
                        error_output = self.process.stderr.readline().strip()
                        if error_output:
                            logger.error(f"Encodec stderr: {error_output}")
                    elif read_stream is self.process.stdout:
                        result = self.process.stdout.readline().strip()
                        logger.debug(f"Received from Encodec process: {result}")
                        if result:
                            if result.startswith(prefix):
                                response_line = result[len(prefix) :]
                                self.process.stdin.write(f"{prefix}ok\n")
                                self.process.stdin.flush()
                                return response_line
                            elif result.startswith("Error:"):
                                # Log other lines but don't treat as the response
                                raise RuntimeError("encodec failed: {}".format(result))
                            else:
                                logger.info(result)

            except Exception as e:
                raise RuntimeError(f"Error reading from Encodec process: {e}")

        if not response_line:
            raise TimeoutError(
                f"Timeout waiting for response from Encodec process for request {uid}"
            )

        # Send simple ack and return response (like DNSMOS implementation)
