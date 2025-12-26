"""
GLM-TTS Inference Server for UltraEval-Audio
"""

import argparse
import json
import logging
import os
import select
import sys
import tempfile
import time

# Add current directory first to import local glmtts_inference.py
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_DIR)

# Add GLM-TTS repo to path (cloned via pre_command)
GLM_TTS_REPO_DIR = "third_party/GLM-TTS"
sys.path.insert(1, GLM_TTS_REPO_DIR)

import torch
import torchaudio

from glmtts_inference import load_models, generate_long, DEVICE
from utils import seed_util

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GLM-TTS Inference Server")
    parser.add_argument(
        "--ckpt_dir", type=str, required=True, help="Path to GLM-TTS ckpt directory"
    )
    parser.add_argument(
        "--use_cache", action="store_true", default=False, help="Use cache"
    )
    parser.add_argument(
        "--use_phoneme", action="store_true", default=False, help="Enable phoneme mode"
    )
    parser.add_argument("--sample_rate", type=int, default=24000, help="Sample rate")
    args = parser.parse_args()

    logger.info(f"Loading GLM-TTS models from {args.ckpt_dir}")
    frontend, text_frontend, speech_tokenizer, llm, flow = load_models(
        ckpt_dir=args.ckpt_dir,
        repo_dir=GLM_TTS_REPO_DIR,
        use_phoneme=args.use_phoneme,
        sample_rate=args.sample_rate,
    )
    logger.info("GLM-TTS models loaded successfully")

    enable_rtf = int(os.environ.get("ENABLE_RTF", "0"))
    logger.info(f"ENABLE_RTF: {enable_rtf}")

    print("GLM-TTS server started. Waiting for input...", flush=True)

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

            text = x["text"]
            prompt_audio = x["prompt_audio"]
            prompt_text = x.get("prompt_text", "")
            seed = x.get("seed", 0)

            with torch.no_grad():
                start_time = time.time()
                seed_util.set_seed(seed)

                # Text normalization
                prompt_text_normalized = (
                    text_frontend.text_normalize(prompt_text) if prompt_text else ""
                )
                synth_text_normalized = text_frontend.text_normalize(text)

                # Extract prompt features
                prompt_text_token = frontend._extract_text_token(
                    prompt_text_normalized + " "
                )
                prompt_speech_token = frontend._extract_speech_token([prompt_audio])
                speech_feat = frontend._extract_speech_feat(
                    prompt_audio, sample_rate=args.sample_rate
                )
                embedding = frontend._extract_spk_embedding(prompt_audio)

                # Prepare cache
                cache_speech_token = [prompt_speech_token.squeeze().tolist()]
                flow_prompt_token = torch.tensor(
                    cache_speech_token, dtype=torch.int32
                ).to(DEVICE)

                cache = {
                    "cache_text": [prompt_text_normalized],
                    "cache_text_token": [prompt_text_token],
                    "cache_speech_token": cache_speech_token,
                    "use_cache": args.use_cache,
                }

                # Generate speech
                tts_speech, _, _, _ = generate_long(
                    frontend=frontend,
                    text_frontend=text_frontend,
                    llm=llm,
                    flow=flow,
                    text_info=["uttid", synth_text_normalized],
                    cache=cache,
                    embedding=embedding,
                    seed=seed,
                    flow_prompt_token=flow_prompt_token,
                    speech_feat=speech_feat,
                    device=DEVICE,
                    use_phoneme=args.use_phoneme,
                )

                end_time = time.time()
                inference_time = end_time - start_time

                # Save to temporary file
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    torchaudio.save(f.name, tts_speech, args.sample_rate)
                    output_path = f.name

                if enable_rtf == 1:
                    audio_duration = tts_speech.shape[1] / args.sample_rate
                    rtf = inference_time / audio_duration if audio_duration > 0 else 0
                    result = json.dumps({"audio": output_path, "RTF": rtf})
                    logger.info(
                        f"RTF: {rtf:.4f} (inference: {inference_time:.2f}s, audio: {audio_duration:.2f}s)"
                    )
                else:
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
