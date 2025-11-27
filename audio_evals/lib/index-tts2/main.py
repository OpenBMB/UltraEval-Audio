import argparse
import json
import logging
import os
import select
import sys
import tempfile
import time
import traceback

from indextts.infer_v2 import IndexTTS2

# Basic logging setup for the server script
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TTS2Processor:
    def __init__(
        self,
        model_dir,
        config_path,
        use_fp16=False,
        use_cuda_kernel=False,
        use_deepspeed=False,
    ):
        """
        Initializes the TTS2 processor.

        Args:
            model_dir: Path to model checkpoints directory
            config_path: Path to config.yaml
            use_fp16: Whether to use fp16 inference
            use_cuda_kernel: Whether to use CUDA kernel
            use_deepspeed: Whether to use deepspeed
        """
        self.model_dir = model_dir
        self.config_path = config_path
        self.use_fp16 = use_fp16
        self.use_cuda_kernel = use_cuda_kernel
        self.use_deepspeed = use_deepspeed

        # 从环境变量获取 ENABLE_RTF 设置，默认为0
        self.enable_rtf = int(os.environ.get("ENABLE_RTF", "0"))
        logger.info(f"ENABLE_RTF: {self.enable_rtf}")

        logger.info(f"Loading TTS2 model from {model_dir} with config {config_path}")
        try:
            self.tts = IndexTTS2(
                cfg_path=config_path,
                model_dir=model_dir,
                use_fp16=use_fp16,
                use_cuda_kernel=use_cuda_kernel,
                use_deepspeed=use_deepspeed,
            )
            logger.info("Successfully loaded TTS2 model")
        except Exception as e:
            import traceback

            traceback.print_exc()
            logger.error(f"Failed to load TTS2 model: {e}")
            raise

    def process_text(self, text, audio_prompt):
        """
        Generates speech from text using the provided audio prompt as reference.

        Args:
            text: Text to synthesize
            audio_prompt: Path to reference audio file (spk_audio_prompt)

        Returns:
            Path to generated audio file or dict with RTF info
        """
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                output_path = f.name

            logger.info(f"Generating speech for text: {text[:50]}...")

            # 记录开始时间用于RTF计算
            start_time = time.time()

            self.tts.infer(
                spk_audio_prompt=audio_prompt,
                text=text,
                output_path=output_path,
                verbose=False,
            )

            # 记录结束时间
            end_time = time.time()
            inference_time = end_time - start_time

            # 根据ENABLE_RTF设置返回不同格式
            if self.enable_rtf == 1:
                # 读取音频文件获取时长信息
                try:
                    import soundfile as sf

                    audio_data, sample_rate = sf.read(output_path)
                    audio_duration = len(audio_data) / sample_rate
                except ImportError:
                    # 如果没有soundfile，使用估算方法
                    # 假设采样率为22050（常用采样率）
                    try:
                        import wave

                        with wave.open(output_path, "rb") as wav_file:
                            frames = wav_file.getnframes()
                            sample_rate = wav_file.getframerate()
                            audio_duration = frames / sample_rate
                    except:
                        # 最后的备用方案，使用文件大小估算
                        file_size = os.path.getsize(output_path)
                        audio_duration = file_size / (2 * 22050)  # 假设16bit, 22050Hz

                # 计算RTF (Real Time Factor)
                rtf = inference_time / audio_duration if audio_duration > 0 else 0

                result = {"audio": output_path, "RTF": rtf}
                logger.info(
                    f"RTF: {rtf:.4f} (inference: {inference_time:.2f}s, audio: {audio_duration:.2f}s)"
                )
                return result
            else:
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
        "--config_path",
        type=str,
        default="audio_evals/lib/index-tts2/checkpoints/config.yaml",
        help="Path to config.yaml file",
    )
    parser.add_argument(
        "--use_fp16", action="store_true", default=False, help="Use fp16 inference"
    )
    parser.add_argument(
        "--use_cuda_kernel", action="store_true", default=False, help="Use CUDA kernel"
    )
    parser.add_argument(
        "--use_deepspeed", action="store_true", default=False, help="Use deepspeed"
    )
    args = parser.parse_args()

    try:
        processor = TTS2Processor(
            model_dir=args.model_dir,
            config_path=args.config_path,
            use_fp16=args.use_fp16,
            use_cuda_kernel=args.use_cuda_kernel,
            use_deepspeed=args.use_deepspeed,
        )
    except Exception as e:
        print(f"Failed to initialize TTS2Processor: {e}", file=sys.stderr, flush=True)
        sys.exit(1)

    print("TTS2 main.py server started. Waiting for input...", flush=True)

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

            result = processor.process_text(text, audio_prompt)

            # 根据返回类型处理输出格式
            if isinstance(result, dict):
                result_json = json.dumps(result)
            else:
                result_json = result

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
