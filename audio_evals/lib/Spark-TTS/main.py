import argparse
import json
import select
import sys
import tempfile

import soundfile as sf
from cli.SparkTTS import SparkTTS
import torch
import logging

logging.basicConfig(level=logging.INFO)
device = "cuda" if torch.cuda.is_available() else "cpu"

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", type=str, required=True, help="Path to checkpoint file"
    )
    parser.add_argument(
        "--vc_mode", action="store_true", default=False, help="Path to checkpoint file"
    )
    config = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    logger.info(f"Loading model from {config.path}")
    model = SparkTTS(config.path, device)
    logger.info(f"successfully loaded model")

    while True:
        try:
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

            with torch.no_grad():
                if config.vc_mode:
                    wav = model.inference(
                        x["text"],
                        prompt_speech_path=x["prompt_audio"],
                        prompt_text=x["prompt_text"],
                    )
                else:
                    wav = model.inference(
                        x["text"],
                        gender=x.get("gender", "female"),
                        pitch=x.get("pitch", "moderate"),
                        speed=x.get("speed", "moderate"),
                    )

                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    sf.write(f.name, wav, samplerate=16000)
                    while True:
                        print(f"{prefix}{f.name}", flush=True)
                        rlist, _, _ = select.select([sys.stdin], [], [], 1)
                        if rlist:
                            finish = sys.stdin.readline().strip()
                            if finish == "{}close".format(prefix):
                                break
                        print("not found close signal, will emit again", flush=True)
        except Exception as e:
            print(f"Error: {str(e)}", flush=True)
