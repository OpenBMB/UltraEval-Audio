import argparse
import json
import logging
import os
import select
import sys
import tempfile
import time

import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", type=str, required=True, help="Path to Qwen3-TTS model"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="custom_voice",
        choices=["custom_voice", "voice_design", "voice_clone"],
        help="Generation mode: custom_voice, voice_design, or voice_clone",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Model dtype",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to run the model on",
    )
    args = parser.parse_args()

    # Determine dtype
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(args.dtype, torch.bfloat16)

    logger.info(f"Loading Qwen3-TTS model from {args.path} with dtype={args.dtype}")
    
    # Try to use flash attention if available
    try:
        model = Qwen3TTSModel.from_pretrained(
            args.path,
            device_map=args.device,
            dtype=dtype,
            attn_implementation="flash_attention_2",
        )
        logger.info("Loaded with flash_attention_2")
    except Exception as e:
        logger.warning(f"Failed to load with flash_attention_2: {e}, falling back to default")
        model = Qwen3TTSModel.from_pretrained(
            args.path,
            device_map=args.device,
            dtype=dtype,
        )
    
    logger.info(f"Qwen3-TTS model loaded successfully in {args.mode} mode")
    
    # Get supported speakers and languages for custom voice mode
    if args.mode == "custom_voice":
        try:
            speakers = model.get_supported_speakers()
            languages = model.get_supported_languages()
            logger.info(f"Supported speakers: {speakers}")
            logger.info(f"Supported languages: {languages}")
        except Exception as e:
            logger.warning(f"Could not get supported speakers/languages: {e}")

    # Enable RTF tracking from environment variable
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
            x = json.loads(prompt[anchor + 2:])

            # Record start time for RTF calculation
            torch.cuda.synchronize()
            start_time = time.time()

            # Extract common parameters
            text = x.pop("text")
            language = x.pop("language", "Auto")
            
            if args.mode == "custom_voice":
                # Custom voice generation
                speaker = x.pop("speaker", "Vivian")
                instruct = x.pop("instruct", None)
                generate_kwargs = {
                    "text": text,
                    "language": language,
                    "speaker": speaker,
                }
                generate_kwargs.update(x)
                if instruct:
                    generate_kwargs["instruct"] = instruct
                logger.info(f"generate_custom_voice kwargs: {generate_kwargs}")
                wavs, sr = model.generate_custom_voice(**generate_kwargs)
                
            elif args.mode == "voice_design":
                # Voice design generation
                instruct = x.pop("instruct", "")
                logger.info(f"voice_design: text: {text}, language: {language}, instruct: {instruct}, **x: {x}")
                wavs, sr = model.generate_voice_design(
                    text=text,
                    language=language,
                    instruct=instruct,
                    **x,
                )
                
            elif args.mode == "voice_clone":
                # Voice clone generation
                ref_audio = x.pop("prompt_audio")
                ref_text = x.pop("prompt_text")
                
                if ref_audio is None:
                    raise ValueError("ref_audio is required for voice_clone mode")
                logger.info(f"ref_audio: {ref_audio}, ref_text: {ref_text}, text: {text}, language: {language}, **x: {x}")
                wavs, sr = model.generate_voice_clone(
                    text=text,
                    language=language,
                    ref_audio=ref_audio,
                    ref_text=ref_text,
                    **x,
                )
            else:
                raise ValueError(f"Unknown mode: {args.mode}")

            # Record end time
            torch.cuda.synchronize()
            end_time = time.time()
            inference_time = end_time - start_time

            # Save output to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                sf.write(f.name, wavs[0], sr)
                output_path = f.name

            # Return result with optional RTF
            if enable_rtf == 1:
                audio_duration = len(wavs[0]) / sr
                rtf = inference_time / audio_duration if audio_duration > 0 else 0
                result = json.dumps({"audio": output_path, "RTF": rtf})
                logger.info(
                    f"RTF: {rtf:.4f} (inference: {inference_time:.2f}s, audio: {audio_duration:.2f}s)"
                )
            else:
                result = output_path

            # Output result with retry mechanism
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
            import traceback
            traceback.print_exc()
            print(f"Error: {str(e)}", flush=True)
