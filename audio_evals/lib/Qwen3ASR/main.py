import argparse
import json
import select
import subprocess
import sys
import tempfile

import torch
from qwen_asr import Qwen3ASRModel


def get_model(
    path,
    dtype="bfloat16",
    device_map="cuda:0",
    max_new_tokens=256,
    max_inference_batch_size=32,
):
    torch_dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
    model = Qwen3ASRModel.from_pretrained(
        path,
        dtype=torch_dtype,
        device_map=device_map,
        max_inference_batch_size=max_inference_batch_size,
        max_new_tokens=max_new_tokens,
    )
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", type=str, required=True, help="Path to checkpoint file"
    )
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--device_map", type=str, default="cuda:0")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--max_inference_batch_size", type=int, default=32)
    config = parser.parse_args()

    model = get_model(
        config.path,
        dtype=config.dtype,
        device_map=config.device_map,
        max_new_tokens=config.max_new_tokens,
        max_inference_batch_size=config.max_inference_batch_size,
    )
    print("Model loaded from checkpoint: {}".format(config.path), flush=True)

    while True:
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
        payload = prompt[anchor + 2 :].strip()
        try:
            try:
                data = json.loads(payload)
            except json.JSONDecodeError:
                data = {"audio": payload}
            audio_path = data["audio"]
            language = data.get("language") or None
            with tempfile.NamedTemporaryFile(suffix=".wav") as wav_file:
                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-i",
                        audio_path,
                        "-ar",
                        "16000",
                        "-ac",
                        "1",
                        "-f",
                        "wav",
                        wav_file.name,
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                results = model.transcribe(audio=wav_file.name, language=language)
                text = results[0].text.replace("\n", " ").strip()
            retry = 3
            while retry:
                retry -= 1
                print("{}{}".format(prefix, text), flush=True)
                rlist, _, _ = select.select([sys.stdin], [], [], 1)
                if rlist:
                    finish = sys.stdin.readline().strip()
                    if finish == "{}close".format(prefix):
                        break
                print("not found close signal, will emit again", flush=True)
        except Exception as e:
            import traceback

            traceback.print_exc()
            print("Error:{}".format(e), flush=True)
