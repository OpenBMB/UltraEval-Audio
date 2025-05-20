import argparse
import os
import select
import sys
import torchaudio

import lightning_module
import torch

device = "cpu"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", type=str, required=True, help="Path to checkpoint file"
    )
    config = parser.parse_args()

    ssl_model = os.path.join(config.path, "wav2vec_small.pt")
    os.environ["SSL_MODEL_PATH"] = ssl_model

    model = (
        lightning_module.BaselineLightningModule.load_from_checkpoint(
            os.path.join(config.path, "epoch=3-step=7459.ckpt"), map_location="cpu"
        )
        .eval()
        .to(device)
    )
    print("Model loaded from checkpoint: {}".format(config.path))

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
        try:
            wav, sr = torchaudio.load(prompt[anchor + 2 :])
            wavs = wav.to(device)
            if len(wavs.shape) == 1:
                wavs = wavs.unsqueeze(0).unsqueeze(0)
            elif len(wavs.shape) == 2:
                wavs = wavs.mean(dim=0, keepdim=True)
                wavs = wavs.unsqueeze(0)
            elif len(wavs.shape) != 3:
                raise ValueError("Dimension of input tensor needs to be <= 3.")

            if sr != 16000:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sr,
                    new_freq=16000,
                    resampling_method="sinc_interpolation",
                    lowpass_filter_width=6,
                    dtype=torch.float32,
                ).to(device)
                wavs = resampler(wavs)

            batch = {
                "wav": wavs,
                "domains": torch.zeros(wavs.size(0), dtype=torch.int).to(device),
                "judge_id": torch.ones(wavs.size(0), dtype=torch.int).to(device) * 288,
            }

            with torch.no_grad():
                output = model(batch)

            retry = 3
            while retry:
                print(
                    "{}{}".format(
                        prefix,
                        output.mean(dim=1).squeeze(1).cpu().detach().item() * 2 + 3,
                    ),
                    flush=True,
                )
                rlist, _, _ = select.select([sys.stdin], [], [], 1)
                if rlist:
                    finish = sys.stdin.readline().strip()
                    if finish == "{}close".format(prefix):
                        break
                print("not found close signal, will emit again", flush=True)
                retry -= 1
        except Exception as e:
            import traceback

            traceback.print_exc()
            print("Error:{}".format(e))
