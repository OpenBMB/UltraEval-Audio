import argparse
import json
import logging
import select
import sys
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
import torch


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", type=str, required=True, help="Path to SenseVoice model"
    )
    config = parser.parse_args()

    # Initialize model
    model = AutoModel(
        model=config.path,
        vad_model="fsmn-vad",
        vad_kwargs={"max_single_segment_time": 30000},
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    logger.info(f"Using SenseVoice model from: {config.path}")

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
            res = model.generate(
                input=x["audio"],
                cache={},
                language=x.get("language", "auto"),
                use_itn=True,
                batch_size_s=30000,
                merge_vad=True,
                merge_length_s=15,
            )
            text = rich_transcription_postprocess(res[0]["text"])
            while True:
                print(f"{prefix}{text}", flush=True)
                rlist, _, _ = select.select([sys.stdin], [], [], 1)
                if rlist:
                    finish = sys.stdin.readline().strip()
                    if finish == "{}close".format(prefix):
                        break
                print("not found close signal, will emit again", flush=True)

        except Exception as e:
            print(f"Error: {str(e)}", flush=True)
