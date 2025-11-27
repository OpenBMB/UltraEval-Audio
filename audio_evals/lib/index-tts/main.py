import argparse
import json
import logging
import os
import select
import sys
import tempfile
import time
import traceback

from indextts.infer import IndexTTS

# Basic logging setup for the server script
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TTSProcessor:
    def __init__(self, model_dir, config_path, fp16=True, cuda_kernel=False):
        """
        Initializes the TTS processor.

        Args:
            model_dir: Path to model checkpoints directory
            config_path: Path to config.yaml
            fp16: Whether to use fp16 inference
            cuda_kernel: Whether to use CUDA kernel for BigVGAN
        """
        self.model_dir = model_dir
        self.config_path = config_path
        self.fp16 = fp16
        self.cuda_kernel = cuda_kernel

        logger.info(f"Loading TTS model from {model_dir} with config {config_path}")
        try:
            self.tts = IndexTTS(
                cfg_path=config_path,
                model_dir=model_dir,
                is_fp16=fp16,
                use_cuda_kernel=cuda_kernel,
            )
            logger.info("Successfully loaded TTS model")
        except Exception as e:
            logger.error(f"Failed to load TTS model: {e}")
            raise

    def process_text(self, text, audio_prompt):
        """
        Generates speech from text using the provided audio prompt as reference.

        Args:
            text: Text to synthesize
            audio_prompt: Path to reference audio file

        Returns:
            Path to generated audio file
        """
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                output_path = f.name

            logger.info(f"Generating speech for text: {text[:50]}...")
            self.tts.infer(
                audio_prompt=audio_prompt, text=text, output_path=output_path
            )
            return output_path
        except Exception as e:
            logger.error(f"Error processing text: {e}")
            traceback.print_exc(file=sys.stderr)
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        type=str,
        default="checkpoints",
        help="Path to model checkpoints directory",
    )
    parser.add_argument(
        "--fp16", action="store_true", default=True, help="Use fp16 inference"
    )
    args = parser.parse_args()

    try:
        processor = TTSProcessor(
            model_dir=args.model_dir,
            config_path="audio_evals/lib/index-tts/checkpoints/config.yaml",
            fp16=args.fp16,
        )
    except Exception as e:
        print(f"Failed to initialize TTSProcessor: {e}", file=sys.stderr, flush=True)
        sys.exit(1)

    print("TTS main.py server started. Waiting for input...", flush=True)

    while True:
        try:
            prompt = input()
            anchor = prompt.find("->")
            if anchor == -1:
                print(
                    f"invalid input format. Expected 'prefix->json_input', got '{prompt}'",
                    file=sys.stderr,
                    flush=True,
                )
                continue

            prefix = prompt[:anchor].strip() + "->"
            input_json_str = prompt[anchor + 2 :].strip()

            input_data = json.loads(input_json_str)
            text = input_data["text"]
            audio_prompt = input_data["prompt_audio"]

            output_filepath = processor.process_text(text, audio_prompt)
            result_json = output_filepath

            # Wait for acknowledgment
            ack_wait_start = time.time()
            while time.time() - ack_wait_start < 60:  # 60s timeout
                print(f"{prefix}{result_json}", flush=True)
                print(
                    f"Sent results for text: {text[:50]}... Waiting for ack...",
                    flush=True,
                )
                rlist, _, xlist = select.select([sys.stdin], [], [sys.stdin], 1.0)
                if rlist:
                    ack_signal = sys.stdin.readline().strip()
                    expected_ack = f"{prefix.strip('->')}->ok"
                    if ack_signal == expected_ack:
                        break
                    else:
                        print(
                            f"Warning: Received unexpected input while waiting for ack for {prefix}: '{ack_signal}'. Expected '{expected_ack}'",
                            file=sys.stderr,
                            flush=True,
                        )
        except Exception as e:
            traceback.print_exc(file=sys.stderr)
            print(f"Error: in main loop: {e}", file=sys.stderr, flush=True)
