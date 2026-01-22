"""
Step-Audio-R1.1 model integration using vLLM with isolated environment.

Step-Audio-R1.1 is a real-time speech model that supports:
- Mind-Paced Speaking (Low Latency)
- Acoustic-Grounded Reasoning (High Intelligence)

Reference: https://huggingface.co/stepfun-ai/Step-Audio-R1.1
"""

import json
import logging
import os
import select
import sys
from typing import Dict, Any, List

# Add the StepAudio library to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../lib/StepAudio"))

from stepaudior1vllm import StepAudioR1 as StepAudioR1Client

from audio_evals.base import PromptStruct
from audio_evals.isolate import isolated
from audio_evals.models.model import OfflineModel

logger = logging.getLogger(__name__)


@isolated(
    "audio_evals/lib/StepAudio/serve.py",
    pre_command="mkdir -p ./third_party && "
    "([ ! -d './third_party/vllm-step-audio' ] && "
    "git clone -b step-audio-2-mini https://github.com/stepfun-ai/vllm.git ./third_party/vllm-step-audio && cd ../../) || true && "
    "(python -c 'import vllm' 2>/dev/null || VLLM_USE_PRECOMPILED=1 uv pip install -e ./third_party/vllm-step-audio)",
)
class StepAudioR1(OfflineModel):
    """
    Step-Audio-R1.1 model using vLLM with isolated environment.

    This model starts a vLLM server in an isolated environment and
    communicates via StepAudioR1 HTTP API.

    The server automatically finds an available port and communicates
    it back via stdout.

    Requirements:
        - Customized vLLM from https://github.com/stepfun-ai/vllm (step-audio-2-mini branch)
        - Step-Audio-R1.1 model weights
    """

    def __init__(
        self,
        model_path: str,
        start_port: int = 9999,
        tensor_parallel_size: int = 4,
        max_model_len: int = 16384,
        max_num_seqs: int = 32,
        gpu_memory_utilization: float = 0.85,
        startup_timeout: int = 600,
        extract_thinking: bool = True,
        speech: bool = False,
        sample_params: Dict[str, Any] = None,
    ):
        """
        Initialize Step-Audio-R1.1 model with isolated vLLM server.

        Args:
            model_path: Path to Step-Audio-R1.1 model weights
            start_port: Starting port to search from (default: 9999)
            tensor_parallel_size: Number of GPUs for tensor parallelism (default: 4)
            max_model_len: Maximum model context length (default: 16384)
            max_num_seqs: Maximum number of sequences (default: 32)
            gpu_memory_utilization: GPU memory utilization (default: 0.85)
            startup_timeout: Timeout for server startup in seconds (default: 600)
            extract_thinking: Whether to extract and remove <think>...</think> blocks
            speech: Whether to enable audio output support (default: False)
            sample_params: Additional sampling parameters
        """
        if not os.path.exists(model_path):
            model_path = self._download_model_from_modelscope(model_path)
        if model_path.endswith("/"):
            model_path = model_path[:-1]
        # Build command args for isolated decorator
        self.command_args = {
            "model_path": model_path,
            "start_port": start_port,
            "tensor_parallel_size": tensor_parallel_size,
            "max_model_len": max_model_len,
            "max_num_seqs": max_num_seqs,
            "gpu_memory_utilization": gpu_memory_utilization,
            "startup_timeout": startup_timeout,
        }
        self.model_name = model_path.split("/")[-1]
        self.extract_thinking = extract_thinking
        self.speech = speech
        self.port = None
        self.client = None
        self._initialized = False

        # Call parent init (this triggers isolated decorator's new_init)
        super().__init__(is_chat=True, sample_params=sample_params)
        # Note: self.process is set by @isolated decorator AFTER this __init__ returns

    def _ensure_initialized(self):
        """Lazy initialization: wait for port and create client on first use."""
        if self._initialized:
            return

        # Wait for server to report port
        self._wait_for_port()

        # Initialize OpenAI client
        api_url = f"http://localhost:{self.port}/v1/chat/completions"
        self.client = StepAudioR1Client(api_url=api_url, model_name=self.model_name)
        logger.info(f"Step-Audio-R1.1 client ready at {api_url}")
        self._initialized = True

    def _wait_for_port(self):
        """Wait for and read the port number from the server process."""
        logger.info("Waiting for server to report port...")

        while True:
            reads, _, _ = select.select(
                [self.process.stdout, self.process.stderr], [], [], 1.0
            )
            for read in reads:
                if read is self.process.stdout:
                    line = self.process.stdout.readline()
                    if line:
                        line = line.strip()
                        if line.startswith("PORT:"):
                            self.port = int(line.split(":")[1])
                            logger.info(f"Server started on port {self.port}")
                            return
                        else:
                            logger.debug(f"stdout: {line}")
                if read is self.process.stderr:
                    error_line = self.process.stderr.readline()
                    if error_line:
                        logger.info(f"stderr: {error_line.strip()}")

            # Check if process has exited
            if self.process.poll() is not None:
                raise RuntimeError("Server process exited before reporting port")

    def _convert_prompt_to_messages(self, prompt: PromptStruct) -> List[Dict]:
        """Convert PromptStruct to StepAudioR1 message format."""
        messages = []

        for item in prompt:
            role = item["role"]
            contents = item.get("contents", [])

            if role == "user":
                role = "human"

            content_list = []
            for content in contents:
                content_type = content.get("type")
                value = content.get("value")

                if content_type == "text":
                    content_list.append({"type": "text", "text": value})
                elif content_type == "audio":
                    content_list.append({"type": "audio", "audio": value})

            if len(content_list) == 1 and content_list[0].get("type") == "text":
                messages.append({"role": role, "content": content_list[0]["text"]})
            else:
                messages.append({"role": role, "content": content_list})

        messages.append({"role": "assistant", "content": "<think>\n", "eot": False})

        return messages

    def _extract_response(self, text: str) -> str:
        """Extract the actual response, removing <think>...</think> blocks."""
        if not self.extract_thinking or not text:
            return text

        return text.split("</think>")[-1].strip()

    def _inference(self, prompt: PromptStruct, **kwargs) -> str:
        """Perform inference with Step-Audio-R1.1."""
        # Lazy initialization on first call
        self._ensure_initialized()

        # Check process is still running
        if not self.check_process_status():
            raise RuntimeError("vLLM server process has exited")

        # Build messages from prompt
        messages = self._convert_prompt_to_messages(prompt)
        logger.debug(
            f"Built messages: {json.dumps(messages, ensure_ascii=False)[:500]}..."
        )

        # Set default parameters
        api_params = {
            "stop_token_ids": [151665],
        }
        api_params.update(kwargs)

        logger.info(f"Calling {self.model_name} API...")

        full_text = ""
        audio_tokens = []

        try:
            for _, text, audio in self.client.stream(messages, **api_params):
                if text:
                    full_text += text
                if audio:
                    audio_tokens.extend(audio)
        except Exception as e:
            logger.error(f"Error during API call: {e}")
            raise

        text_result = self._extract_response(full_text) if full_text else ""
        logger.info(
            f"Extracted response: {text_result[:200] if text_result else 'None'}..."
        )

        if not self.speech:
            return text_result

        result = {"text": text_result}

        if audio_tokens:
            logger.info(f"Received {len(audio_tokens)} audio tokens")

        return json.dumps(result, ensure_ascii=False)
