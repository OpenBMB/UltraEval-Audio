import argparse
import json
import logging
import select
import sys
import torch
import soundfile as sf
import scipy
import whisper


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True, help="Path to Whisper model")
    config = parser.parse_args()

    # Initialize model
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model(config.path).to(device)
    model.eval()
    logger.info(f"Using Whisper model from: {config.path} on device: {device}")
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
            # Process input
            logger.info(f"Received input: {x}")

            result = model.transcribe(x["audio"], language=x.get("language", "english"))
            transcription = result["text"].strip()
            result = {"text": transcription}
            retry = 3
            while retry:
                print(f"{prefix}{result['text']}", flush=True)
                rlist, _, _ = select.select([sys.stdin], [], [], 1)
                if rlist:
                    finish = sys.stdin.readline().strip()
                    if finish == "{}close".format(prefix):
                        break
                print("not found close signal, will emit again", flush=True)
                retry -= 1
        except Exception as e:
            print(f"Error: {str(e)}", flush=True)
