import argparse
import json
import logging
import select
import sys
import dolphin
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--name",
        type=str,
        default="small",
        help="Model name (base, small) or path to model directory",
    )
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Model name (base, small) or path to model directory",
    )
    config = parser.parse_args()

    # Initialize model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = dolphin.load_model(config.name, config.path, device)
    logger.info(f"Using Dolphin model: {config.path} on device: {device}")

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
            kwargs = x.pop("kwargs", {})
            x.update(kwargs)
            logger.info(f"Received input: {x}")

            # Load audio and perform inference
            waveform = dolphin.load_audio(x["audio"])
            result = model(waveform, **kwargs)

            retry = 3
            while retry:
                print(f"{prefix}{result.text}", flush=True)
                rlist, _, _ = select.select([sys.stdin], [], [], 1)
                if rlist:
                    finish = sys.stdin.readline().strip()
                    if finish == "{}close".format(prefix):
                        break
                print("not found close signal, will emit again", flush=True)
                retry -= 1
        except Exception as e:
            print(f"Error: {str(e)}", flush=True)
