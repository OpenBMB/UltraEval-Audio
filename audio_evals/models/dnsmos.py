import logging
import os
import select
import subprocess
import sys
import uuid
import time
from typing import Dict, Tuple

from audio_evals.base import PromptStruct
from audio_evals.constants import DEFAULT_MODEL_PATH
from audio_evals.models.model import OfflineModel
from audio_evals.isolate import isolated

logger = logging.getLogger(__name__)


@isolated("audio_evals/lib/DNSMOS/main.py")
class DNSMOS(OfflineModel):
    """
    Client for interacting with the DNSMOS evaluation script.
    """

    def __init__(
        self,
        model_path: str = "",
        p_model_path: str = "",
        p808_model_path: str = "",
        sample_params: Dict = None,
        *args,
        **kwargs,
    ):
        """
        Initializes the DNSMOS client.

        Args:
            model_path (str): Path to the main DNSMOS ONNX model (sig_bak_ovr.onnx).
            p_model_path (str): Path to the personalized DNSMOS ONNX model.
            p808_model_path (str): Path to the P.808 ONNX model (model_v8.onnx).
            sample_params (Dict, optional): Sampling parameters. Defaults to None.
        """
        if any([not model_path, not p_model_path, not p808_model_path]):
            model_path, p_model_path, p808_model_path = self._download_model(
                DEFAULT_MODEL_PATH
            )

        self.command_args = {
            "model_path": model_path,
            "p_model_path": p_model_path,
            "p808_model_path": p808_model_path,
        }
        # Include any other necessary arguments passed via kwargs for the isolated script
        self.command_args.update(kwargs)

        # DNSMOS returns a dictionary, not just a string, so is_chat should be False
        super().__init__(is_chat=False, sample_params=sample_params)
        logger.info(
            f"DNSMOS client initialized with models: {model_path}, {p_model_path}, {p808_model_path}"
        )

    @staticmethod
    def _download_model(save_path: str) -> Tuple[str]:
        repo_dir = os.path.join(save_path, "DNS-Challenge")
        p835_path = os.path.join(repo_dir, "DNSMOS/DNSMOS/sig_bak_ovr.onnx")
        p835_ppath = os.path.join(repo_dir, "DNSMOS/pDNSMOS/sig_bak_ovr.onnx")
        p808_path = os.path.join(repo_dir, "DNSMOS/DNSMOS/model_v8.onnx")
        if not os.path.exists(repo_dir):
            os.makedirs(repo_dir)
        else:
            if all(
                [
                    os.path.exists(p835_path),
                    os.path.exists(p835_ppath),
                    os.path.exists(p808_path),
                ]
            ):
                return p835_path, p835_ppath, p808_path

        # URL of the repository
        repo_url = "https://github.com/microsoft/DNS-Challenge.git"

        try:
            # Clone the repository
            logger.info(f"Cloning repository from {repo_url}...")
            subprocess.run(["git", "clone", repo_url, repo_dir], check=True)

            logger.info("Repository downloaded successfully!")
            logger.info(f"Repository is located at: {repo_dir}")

        except subprocess.CalledProcessError as e:
            logger.error(f"Error occurred while cloning the repository: {e}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            sys.exit(1)

        return p835_path, p835_ppath, p808_path

    def _inference(self, prompt: PromptStruct, **kwargs) -> str:
        """
        Sends an audio file path to the DNSMOS script and returns the raw response string.

        Args:
            prompt (PromptStruct): A dictionary containing the audio file path under the key 'audio'.
            **kwargs: Additional keyword arguments (not used in this implementation).

        Returns:
            str: The raw response string received from the DNSMOS script (expected to be JSON).

        Raises:
            RuntimeError: If the DNSMOS script returns an error or times out.
        """
        audio_filepath = prompt["audio"]
        if not isinstance(audio_filepath, str):
            raise TypeError(
                f"Expected audio filepath to be a string, got {type(audio_filepath)}"
            )

        uid = str(uuid.uuid4())
        prefix = f"{uid}->"
        request = f"{prefix}{audio_filepath}\n"

        logger.debug(f"Sending request to DNSMOS process: {request.strip()}")

        # Send request using select to avoid blocking indefinitely
        try:
            _, wlist, xlist = select.select(
                [], [self.process.stdin], [self.process.stdin], 60
            )  # 60s timeout
            if xlist:
                raise RuntimeError("DNSMOS stdin is broken (reported by select).")
            if not wlist:
                raise TimeoutError("Timeout waiting for DNSMOS stdin to be writable.")
            self.process.stdin.write(request)
            self.process.stdin.flush()
        except BrokenPipeError:
            raise RuntimeError("DNSMOS process stdin pipe is broken.")
        except Exception as e:
            raise RuntimeError(f"Error writing to DNSMOS process stdin: {e}")

        # Receive response with timeout
        max_wait_time = 120  # seconds
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
                        "DNSMOS stdout/stderr is broken (reported by select)."
                    )

                for read_stream in reads:
                    if read_stream is self.process.stderr:
                        error_output = self.process.stderr.readline().strip()
                        if error_output:
                            logger.error(f"DNSMOS stderr: {error_output}")
                            # Continue reading stdout in case a result is still produced
                            # Or raise immediately: raise RuntimeError(f"DNSMOS error: {error_output}")

                    if read_stream is self.process.stdout:
                        result = self.process.stdout.readline().strip()
                        logger.debug(f"Received from DNSMOS process: {result}")
                        if result:
                            if result.startswith(prefix):
                                ack_signal = f"{prefix}ok\n"
                                self.process.stdin.write(ack_signal)
                                self.process.stdin.flush()
                                response_line = result[len(prefix) :]
                                logger.debug(
                                    f"Found matching response: {response_line}"
                                )
                                return response_line
                            elif "Error:" in result:
                                # Handle errors reported by the script itself in stdout
                                logger.error(f"DNSMOS script reported error: {result}")
                                raise RuntimeError(f"DNSMOS script error: {result}")
                            else:
                                # Log other stdout lines (e.g., debug messages from main.py)
                                logger.info(f"DNSMOS stdout (non-matching): {result}")

            except Exception as e:
                raise RuntimeError(f"Error reading from DNSMOS process: {e}")

        raise TimeoutError(
            f"Timeout waiting for response from DNSMOS process for request {uid}"
        )
