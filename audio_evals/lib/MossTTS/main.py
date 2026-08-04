import argparse
import json
import logging
import os
import select
import sys
import tempfile

import numpy as np
import soundfile as sf
import torch
import torchaudio
from transformers import AutoModel, AutoProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# torchaudio's `.load()`/`.save()` in this version unconditionally delegate to
# torchcodec, which dlopen's libavutil/libavcodec/etc. at runtime. This
# container only ships FFmpeg 4.2 (Ubuntu focal's apt repo has nothing newer)
# and its Python was built without --enable-shared (no libpythonX.Y.so
# anywhere on disk), so torchcodec cannot load under any FFmpeg major version
# it supports (4-8) - this isn't fixable via package installs or
# LD_LIBRARY_PATH. MOSS-TTS's own trust_remote_code processor calls
# torchaudio.load() to read reference audio for voice cloning, so we replace
# both functions with soundfile-based equivalents before any such call runs.


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

# Official MOSS-VoiceGenerator / MossTTSDelay demos disable broken cuDNN SDPA.
if torch.cuda.is_available():
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", type=str, required=True, help="Path or HF repo id of MOSS-TTS model"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Model dtype",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to run the model on",
    )
    args = parser.parse_args()

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(args.dtype, torch.bfloat16)

    logger.info(f"Loading MOSS-TTS model from {args.path} with dtype={args.dtype}")

    # VoiceGenerator recommends normalize_inputs=True; other family checkpoints
    # may ignore or reject the kwarg, so fall back cleanly.
    try:
        processor = AutoProcessor.from_pretrained(
            args.path, trust_remote_code=True, normalize_inputs=True
        )
        logger.info("Loaded processor with normalize_inputs=True")
    except TypeError:
        processor = AutoProcessor.from_pretrained(args.path, trust_remote_code=True)
        logger.info("Loaded processor without normalize_inputs")
    if hasattr(processor, "audio_tokenizer"):
        processor.audio_tokenizer = processor.audio_tokenizer.to(args.device)

    # This model's `attn_implementation="flash_attention_2"` only fails at
    # first actual attention call (flash_attn's import guard is lazy, not
    # checked at from_pretrained time), so a try/except around loading can't
    # catch it. flash_attn is hard to install in this container, so we just
    # never request it and use sdpa instead, which this model's remote code
    # explicitly supports as a non-flash fallback path. MOSS-TTS-Local-
    # Transformer additionally has its own `local_transformer_attn_implementation`
    # config field (defaulting to whatever `attn_implementation` resolves to,
    # but overridable independently), so it's set explicitly too; on the plain
    # MOSS-TTS config this kwarg is simply unrecognized and silently dropped.
    try:
        model = AutoModel.from_pretrained(
            args.path,
            trust_remote_code=True,
            dtype=dtype,
            device_map=args.device,
            attn_implementation="sdpa",
            local_transformer_attn_implementation="sdpa",
        )
        logger.info("Loaded with sdpa attention")
    except Exception as e:
        logger.warning(f"Failed to load with sdpa: {e}, falling back to default")
        model = AutoModel.from_pretrained(
            args.path,
            trust_remote_code=True,
            dtype=dtype,
            device_map=args.device,
        )
    model.eval()

    logger.info("MOSS-TTS model loaded successfully")

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
            x.pop("prompt_text", None)
            instruction = x.pop("instruction", None)
            language = x.pop("language", None)
            tokens = x.pop("tokens", None)
            max_new_tokens = x.pop("max_new_tokens", 4096)

            message_kwargs = {"text": text}
            if prompt_audio:
                message_kwargs["reference"] = [prompt_audio]
            if instruction:
                message_kwargs["instruction"] = instruction
            if language:
                message_kwargs["language"] = language
            if tokens:
                message_kwargs["tokens"] = tokens

            logger.info(f"build_user_message kwargs: {message_kwargs}")
            message = processor.build_user_message(**message_kwargs)

            # Official MossTTSDelay / VoiceGenerator API expects a batch of
            # conversations: List[List[message]].
            inputs = processor([[message]], mode="generation")
            inputs = {
                k: v.to(model.device) if hasattr(v, "to") else v
                for k, v in inputs.items()
            }

            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, **x)

            decoded = processor.decode(outputs)
            audio = decoded[0].audio_codes_list[0]

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                output_path = f.name
            torchaudio.save(
                output_path,
                audio.unsqueeze(0) if audio.dim() == 1 else audio,
                processor.model_config.sampling_rate,
            )

            result = output_path

            retry = 3
            while retry:
                retry -= 1
                print(f"{prefix}{result}", flush=True)
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
