import argparse
import json
import logging
import os
import select
import sys
import tempfile
import time

import soundfile as sf
from voxcpm import VoxCPM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", type=str, required=True, help="Path to VoxCPM2 model directory"
    )
    parser.add_argument("--denoise", action="store_true", help="Enable denoising")
    parser.add_argument(
        "--denoise_path",
        type=str,
        required=False,
        default="iic/speech_zipenhancer_ans_multiloss_16k_base",
        help="Path to denoising model",
    )
    args = parser.parse_args()

    logger.info(f"Loading VoxCPM2 model from {args.path}, denoise: {args.denoise}")

    model = VoxCPM.from_pretrained(
        args.path, load_denoiser=args.denoise, zipenhancer_model_id=args.denoise_path
    )
    sample_rate = model.tts_model.sample_rate
    logger.info(f"VoxCPM2 successfully loaded, sample_rate={sample_rate}")

    enable_rtf = int(os.environ.get("ENABLE_RTF", "0"))
    logger.info(f"ENABLE_RTF: {enable_rtf}")

    while True:
        try:
            prompt = input()
            anchor = prompt.find("->")
            if anchor == -1:
                print(
                    f"Error: Invalid conversation format, must contain '->', but got {prompt}",
                    flush=True,
                )
                continue

            prefix = prompt[:anchor].strip() + "->"
            x = json.loads(prompt[anchor + 2 :])

            start_time = time.time()

            text = x.pop("text")
            prompt_audio = x.pop("prompt_audio", None)
            prompt_text = x.pop("prompt_text", None)
            reference_audio = x.pop("reference_audio", None)

            # Route audio inputs to VoxCPM2 API:
            # - reference_wav_path: isolated voice cloning (no transcript needed)
            # - prompt_wav_path + prompt_text: continuation mode (both required)
            # - both together: ultimate cloning (reference isolation + continuation)
            if prompt_audio is not None and reference_audio is None:
                reference_audio = prompt_audio
                if prompt_text is None:
                    # No transcript → use only as reference (controllable voice cloning)
                    prompt_audio = None

            wav = model.generate(
                text=text,
                prompt_wav_path=prompt_audio,
                prompt_text=prompt_text,
                reference_wav_path=reference_audio,
                **x,
            )

            end_time = time.time()
            inference_time = end_time - start_time

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                sf.write(f.name, wav, samplerate=sample_rate)
                output_path = f.name

            if enable_rtf == 1:
                audio_duration = len(wav) / sample_rate
                rtf = inference_time / audio_duration if audio_duration > 0 else 0
                result = json.dumps({"audio": output_path, "RTF": rtf})
                logger.info(
                    f"RTF: {rtf:.4f} (inference: {inference_time:.2f}s, audio: {audio_duration:.2f}s)"
                )
            else:
                result = output_path

            retry = 3
            while retry:
                retry -= 1
                print(f"{prefix}{result}", flush=True)
                rlist, _, _ = select.select([sys.stdin], [], [], 1)
                if rlist:
                    finish = sys.stdin.readline().strip()
                    if finish == f"{prefix}close":
                        break
                print("not found close signal, will emit again", flush=True)

        except Exception as e:
            print(f"Error: {str(e)}", flush=True)
