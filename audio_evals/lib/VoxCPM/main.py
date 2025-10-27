import argparse
import json
import logging
import select
import sys
import tempfile

import torch
import soundfile as sf
from voxcpm import VoxCPM 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def to_numpy(wav):
    # support torch.Tensor / list / numpy
    try:
        import numpy as np
    except Exception:
        np = None

    if "torch" in str(type(wav)):
        wav = wav.detach().cpu().float().numpy()
    elif np is not None and isinstance(wav, np.ndarray):
        wav = wav.astype("float32")
    elif isinstance(wav, list):
        if np is None:
            raise RuntimeError("numpy is required to handle list waveform")
        import numpy as np
        wav = np.asarray(wav, dtype="float32")
    return wav


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--vc_mode", action="store_true", default=False, help="Enable voice clone mode")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    logger.info(f"Loading VoxCPM model from {args.path}")

    model = VoxCPM.from_pretrained(args.path)
    logger.info("VoxCPM successfully loaded")

    while True:
        try:
            prompt = input()
            anchor = prompt.find("->")
            if anchor == -1:
                print(f"Error: Invalid conversation format, must contain '->', but got {prompt}", flush=True)
                continue

            prefix = prompt[:anchor].strip() + "->"
            x = json.loads(prompt[anchor + 2 :])

            with torch.no_grad():
                if args.vc_mode:
                    wav = model.generate(
                        text=x["text"],
                        prompt_wav_path=x["prompt_audio"],
                        prompt_text=x.get("prompt_text", None),
                        denoise=True
                    )
                else:
                    wav = model.generate(
                        text=x["text"]
                    )

                wav = to_numpy(wav)

                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    sf.write(f.name, wav, samplerate=16000)

                retry = 3
                while retry:
                    retry -= 1
                    print(f"{prefix}{f.name}", flush=True)
                    rlist, _, _ = select.select([sys.stdin], [], [], 1)
                    if rlist:
                        finish = sys.stdin.readline().strip()
                        if finish == f"{prefix}close":
                            break
                    print("not found close signal, will emit again", flush=True)

        except Exception as e:
            print(f"Error: {str(e)}", flush=True)