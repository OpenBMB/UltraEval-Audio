import argparse
import json
import os
import select
import subprocess
import sys
import traceback

import numpy as np
import soundfile as sf

TARGET_SAMPLE_RATE = 16000
NATIVE_ROOT_FILES = [
    "chat_template.jinja",
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
]


def load_audio_16k(audio_path: str) -> np.ndarray:
    try:
        samples, sr = sf.read(audio_path, dtype="float32", always_2d=False)
    except (sf.LibsndfileError, RuntimeError):
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            audio_path,
            "-f",
            "f32le",
            "-ac",
            "1",
            "-ar",
            str(TARGET_SAMPLE_RATE),
            "pipe:1",
        ]
        proc = subprocess.run(cmd, capture_output=True, check=False)
        if proc.returncode != 0 or not proc.stdout:
            err = proc.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"ffmpeg decode failed for {audio_path}: {err}")
        samples = np.frombuffer(proc.stdout, dtype=np.float32).copy()
        sr = TARGET_SAMPLE_RATE

    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    if sr != TARGET_SAMPLE_RATE:
        try:
            import soxr

            samples = soxr.resample(samples, sr, TARGET_SAMPLE_RATE).astype(
                np.float32, copy=False
            )
        except ImportError:
            import librosa

            samples = librosa.resample(
                samples, orig_sr=sr, target_sr=TARGET_SAMPLE_RATE
            ).astype(np.float32, copy=False)
    return samples


def get_model(
    path: str,
    model_revision: str = "8c2a8871f1304aa2f634da4fb95e24438c4945b9",
    dtype: str = "bfloat16",
    device_map: str = "cuda:0",
):
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    local_dir = path
    if not os.path.isdir(path):
        local_dir = snapshot_download(
            repo_id=path,
            revision=model_revision,
            allow_patterns=NATIVE_ROOT_FILES,
        )

    torch_dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
    processor = AutoProcessor.from_pretrained(local_dir, local_files_only=True)
    model = AutoModelForMultimodalLM.from_pretrained(
        local_dir,
        dtype=torch_dtype,
        local_files_only=True,
    )
    model.to(device_map)
    model.eval()
    return model, processor


def transcribe(
    model,
    processor,
    audio_path: str,
    language=None,
    max_new_tokens: int = 512,
) -> str:
    import contextlib

    try:
        import torch

        infer_ctx = torch.inference_mode()
    except ImportError:
        infer_ctx = contextlib.nullcontext()

    wav = load_audio_16k(audio_path)
    req_kwargs = {"audio": [wav]}
    if language:
        req_kwargs["language"] = language
    inputs = processor.apply_transcription_request(**req_kwargs)
    prompt_len = inputs["input_ids"].shape[1]
    inputs = inputs.to(model.device, dtype=model.dtype)

    with infer_ctx:
        pred_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    decoded = processor.decode(
        pred_ids[:, prompt_len:], return_format="transcription_only"
    )
    if not decoded:
        raise RuntimeError("OrukFourierASR returned no result")
    text = decoded[0] if isinstance(decoded, list) else str(decoded)
    return " ".join(str(text).split())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True)
    parser.add_argument(
        "--model_revision", default="8c2a8871f1304aa2f634da4fb95e24438c4945b9"
    )
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device_map", default="cuda:0")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--language", default="en")
    config = parser.parse_args()

    model, processor = get_model(
        path=config.path,
        model_revision=config.model_revision,
        dtype=config.dtype,
        device_map=config.device_map,
    )
    print(f"Model loaded from checkpoint: {config.path}", flush=True)

    for raw_prompt in sys.stdin:
        raw_prompt = raw_prompt.rstrip("\n")
        anchor = raw_prompt.find("->")
        if anchor == -1:
            print("Error: Invalid request format; expected '<id>-><json>'", flush=True)
            continue

        prefix = raw_prompt[:anchor].strip() + "->"
        try:
            request = json.loads(raw_prompt[anchor + 2 :].strip())
            text = transcribe(
                model,
                processor,
                audio_path=request["audio"],
                language=request.get("language") or config.language or None,
                max_new_tokens=config.max_new_tokens,
            )
            print(f"{prefix}{text}", flush=True)
            readable, _, _ = select.select([sys.stdin], [], [], 1)
            if readable:
                acknowledgement = sys.stdin.readline().strip()
                if acknowledgement != f"{prefix}close":
                    print(
                        "Worker received an unexpected acknowledgement",
                        file=sys.stderr,
                    )
        except Exception as error:  # noqa: BLE001
            traceback.print_exc(file=sys.stderr)
            print(f"Error:{error}", flush=True)


if __name__ == "__main__":
    main()
