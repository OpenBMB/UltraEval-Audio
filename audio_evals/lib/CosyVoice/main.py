import argparse
import logging
import os
import time
import select
import sys
import tempfile
import torch
import torchaudio
from cosyvoice.cli.cosyvoice import CosyVoice2
import json
from cosyvoice.utils.file_utils import load_wav

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", type=str, required=True, help="Path to CosyVoice model directory"
    )
    parser.add_argument(
        "--vc_mode",
        action="store_true",
        default=False,
        help="Path to CosyVoice model directory",
    )
    config = parser.parse_args()

    logger.info(f"Loading CosyVoice model from {config.path}")
    model = CosyVoice2(config.path, load_jit=False, load_trt=False, fp16=False)
    prompt_speech_16k = load_wav(
        "audio_evals/lib/CosyVoice/asset/zero_shot_prompt.wav", 16000
    )
    logger.info("CosyVoice model loaded")

    # 从环境变量获取 ENABLE_RTF 设置，默认为0
    enable_rtf = int(os.environ.get("ENABLE_RTF", "0"))
    logger.info(f"ENABLE_RTF: {enable_rtf}")

    while True:
        try:
            # Read audio path from stdin
            prompt = input()
            anchor = prompt.find("->")
            if anchor == -1:
                print(
                    "Error: Invalid conversation format, must contains  ->, but {}".format(
                        prompt
                    ),
                    flush=True,
                )
                continue
            prefix = prompt[:anchor].strip() + "->"
            x = json.loads(prompt[anchor + 2 :])

            # 记录开始时间用于RTF计算
            start_time = time.time()
            if config.vc_mode:
                for k in ["text", "prompt_text", "prompt_audio"]:
                    assert k in x, "{} should be input, but {}".format(k, x)
                # Process audio using CosyVoice
                results = model.inference_zero_shot(
                    x["text"],  # Placeholder text
                    x["prompt_text"],  # Placeholder style
                    load_wav(x["prompt_audio"], 16000),
                    stream=False,
                )
            else:
                results = model.inference_cross_lingual(
                    x["text"], prompt_speech_16k, stream=False  # Placeholder text
                )
            res = torch.concat([item["tts_speech"] for item in results], dim=1)

            # 记录结束时间
            end_time = time.time()
            inference_time = end_time - start_time
            # Save output to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                torchaudio.save(f.name, res, model.sample_rate)

                # 根据ENABLE_RTF设置返回不同格式
                if enable_rtf == 1:
                    # 计算音频时长
                    audio_duration = res.shape[1] / model.sample_rate
                    # 计算RTF (Real Time Factor)
                    rtf = inference_time / audio_duration if audio_duration > 0 else 0
                    print(f"rtf: {rtf}", flush=True)
                    result = {"audio": f.name, "RTF": rtf}
                    logger.info(
                        f"RTF: {rtf:.4f} (inference: {inference_time:.2f}s, audio: {audio_duration:.2f}s)"
                    )
                    output_str = json.dumps(result)
                else:
                    output_str = f.name

                while True:
                    print(f"{prefix}{output_str}", flush=True)
                    rlist, _, _ = select.select([sys.stdin], [], [], 1)
                    if rlist:
                        finish = sys.stdin.readline().strip()
                        if finish == "{}close".format(prefix):
                            break
                    print("not found close signal, will emit again", flush=True)

        except Exception as e:
            logger.error(f"Error processing audio: {str(e)}")
            print(f"Error:{str(e)}", flush=True)
