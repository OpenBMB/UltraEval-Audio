import argparse
import json
import logging
import select
import sys
import tempfile

import numpy as np
import soundfile as sf
import torch
import torchaudio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Defensive patch, same rationale as audio_evals/lib/MossTTS/main.py: this
# container's torchaudio.load/save delegate to torchcodec, which cannot
# dlopen FFmpeg/libpython here. dots.tts itself does audio I/O via
# librosa/soundfile, but torchaudio is a hard dependency of the package, so
# patch it before anything can call into it.


def _load_via_soundfile(path, *args, **kwargs):
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return torch.from_numpy(data.T.copy()), sr


def _save_via_soundfile(path, src, sample_rate, *args, **kwargs):
    arr = (
        src.detach().to("cpu").float().numpy()
        if isinstance(src, torch.Tensor)
        else np.asarray(src, dtype="float32")
    )
    if arr.ndim == 2:
        arr = arr.T
        if arr.shape[1] == 1:
            arr = arr[:, 0]
    sf.write(str(path), arr, sample_rate)


torchaudio.load = _load_via_soundfile
torchaudio.save = _save_via_soundfile

# Only these kwargs are forwarded to DotsTtsRuntime.generate(); anything else
# coming from prompt templates or sample_params is dropped instead of leaking
# into the call as an unexpected kwarg.
_GENERATE_KWARGS = {
    "language",
    "speaker_scale",
    "ode_method",
    "num_steps",
    "guidance_scale",
    "normalize_text",
    "template_name",
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", type=str, required=True, help="Path or HF repo id of dots.tts model"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Model precision",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to run the model on",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Enable torch.compile acceleration (one-time warmup at load)",
    )
    args = parser.parse_args()

    if torch.cuda.is_available() and args.device.startswith("cuda"):
        torch.cuda.set_device(args.device)

    from dots_tts.runtime import DotsTtsRuntime

    logger.info(
        f"Loading dots.tts runtime from {args.path} "
        f"(precision={args.dtype}, optimize={args.optimize})"
    )
    runtime = DotsTtsRuntime.from_pretrained(
        args.path,
        precision=args.dtype,
        optimize=args.optimize,
    )
    # Fixed default seed for reproducibility; a per-request "seed" overrides it.
    torch.manual_seed(42)
    logger.info(f"dots.tts runtime loaded, sample_rate={runtime.sample_rate}")

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

            text = x.pop("text")
            prompt_audio = x.pop("prompt_audio", None)
            prompt_text = x.pop("prompt_text", None)
            # voice-clone-en/zh templates carry "lang"; the runtime calls it
            # "language" and normalizes codes/names itself.
            lang = x.pop("lang", None)
            if lang and "language" not in x:
                x["language"] = lang
            seed = x.pop("seed", None)
            if seed is not None:
                torch.manual_seed(int(seed))

            generate_kwargs = {k: v for k, v in x.items() if k in _GENERATE_KWARGS}
            logger.info(
                f"generate: text_len={len(text)} prompt_audio={prompt_audio} "
                f"kwargs={generate_kwargs}"
            )
            result = runtime.generate(
                text=text,
                prompt_audio_path=prompt_audio,
                prompt_text=prompt_text,
                **generate_kwargs,
            )

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                output_path = f.name
            audio = result["audio"].detach().float().cpu().squeeze().numpy()
            sf.write(output_path, audio, int(result["sample_rate"]))

            retry = 3
            while retry:
                retry -= 1
                print(f"{prefix}{output_path}", flush=True)
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
