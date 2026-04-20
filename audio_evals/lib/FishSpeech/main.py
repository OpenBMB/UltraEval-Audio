# pyright: reportMissingImports=false
import argparse
import json
import logging
import os
import select
import sys
import tempfile
import time

import soundfile as sf
import torch

from fish_speech.models.text2semantic.inference import (
    decode_to_audio,
    encode_audio,
    generate_long,
    init_model,
    load_codec_model,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _default_precision(device: str) -> torch.dtype:
    return torch.bfloat16 if device.startswith("cuda") else torch.float32


class FishSpeechProcessor:
    def __init__(self, path: str, compile: bool = False):
        self.path = path
        self.device = _default_device()
        self.precision = _default_precision(self.device)
        self.compile = compile

        logger.info(
            "Loading Fish Audio S2 Pro from %s on %s with precision=%s compile=%s",
            path,
            self.device,
            self.precision,
            compile,
        )

        self.model, self.decode_one_token = init_model(
            path, self.device, self.precision, compile=compile
        )
        with torch.device(self.device):
            self.model.setup_caches(
                max_batch_size=1,
                max_seq_len=self.model.config.max_seq_len,
                dtype=next(self.model.parameters()).dtype,
            )

        codec_checkpoint = os.path.join(path, "codec.pth")
        self.codec = load_codec_model(codec_checkpoint, self.device, self.precision)
        logger.info("Fish Audio S2 Pro loaded successfully")

    def process(self, payload: dict, enable_rtf: bool = False):
        text = payload.pop("text")
        prompt_audio = payload.pop("prompt_audio", None)
        prompt_text = payload.pop("prompt_text", None)
        max_new_tokens = int(payload.pop("max_new_tokens", 0))
        top_p = float(payload.pop("top_p", 0.9))
        top_k = int(payload.pop("top_k", 30))
        temperature = float(payload.pop("temperature", 1.0))
        chunk_length = int(payload.pop("chunk_length", 300))
        iterative_prompt = bool(payload.pop("iterative_prompt", True))

        if prompt_audio and not prompt_text:
            raise ValueError(
                "Fish Audio S2 voice cloning requires both `prompt_audio` and `prompt_text`."
            )

        prompt_tokens = None
        prompt_texts = None
        if prompt_audio:
            prompt_tokens = [encode_audio(prompt_audio, self.codec, self.device).cpu()]
            prompt_texts = [prompt_text]

        if torch.cuda.is_available() and self.device.startswith("cuda"):
            torch.cuda.synchronize()
        start_time = time.time()

        generator = generate_long(
            model=self.model,
            device=self.device,
            decode_one_token=self.decode_one_token,
            text=text,
            num_samples=1,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            compile=self.compile,
            iterative_prompt=iterative_prompt,
            chunk_length=chunk_length,
            prompt_text=prompt_texts,
            prompt_tokens=prompt_tokens,
        )

        codes = []
        for response in generator:
            if response.action == "sample":
                codes.append(response.codes)

        if not codes:
            raise RuntimeError("Fish Audio S2 returned no audio codes")

        merged_codes = torch.cat(codes, dim=1)
        audio = decode_to_audio(merged_codes.to(self.device), self.codec)

        if torch.cuda.is_available() and self.device.startswith("cuda"):
            torch.cuda.synchronize()
        inference_time = time.time() - start_time

        audio_np = audio.float().cpu().numpy()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio_np, self.codec.sample_rate)
            output_path = f.name

        if enable_rtf:
            audio_duration = len(audio_np) / self.codec.sample_rate
            rtf = inference_time / audio_duration if audio_duration > 0 else 0
            return {"audio": output_path, "RTF": rtf}
        return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True, help="Fish Audio S2 path")
    parser.add_argument(
        "--compile", action="store_true", default=False, help="Enable torch.compile"
    )
    args = parser.parse_args()

    try:
        processor = FishSpeechProcessor(
            path=args.path,
            compile=args.compile,
        )
    except Exception as e:
        print(
            f"Failed to initialize FishSpeechProcessor: {e}",
            file=sys.stderr,
            flush=True,
        )
        raise

    enable_rtf = int(os.environ.get("ENABLE_RTF", "0")) == 1
    logger.info("FishSpeech main.py server started. Waiting for input...")

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
            payload = json.loads(prompt[anchor + 2 :])
            result = processor.process(payload, enable_rtf=enable_rtf)

            result_json = json.dumps(result) if isinstance(result, dict) else result

            retry = 3
            while retry:
                retry -= 1
                print(f"{prefix}{result_json}", flush=True)
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
