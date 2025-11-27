import argparse
import json
import logging
import select
import sys

sys.path.append("third_party/MGM-Omni/third_party/Matcha-TTS")
sys.path.append("third_party/MGM-Omni/third_party")
import tempfile
import copy

import torch
import soundfile as sf
import librosa
import whisper
from transformers import TextStreamer

from mgm.constants import DEFAULT_SPEECH_TOKEN, AUDIO_START, AUDIO_END, AUDIO_SEP
from mgm.conversation import conv_templates
from mgm.model.builder import load_pretrained_model
from mgm.utils import disable_torch_init
from mgm.mm_utils import tokenizer_speech_token
from mgm.serve.utils import whispers_asr

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def to_numpy(wav):
    """Convert audio to numpy array format."""
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
    parser.add_argument(
        "--path", type=str, required=True, help="Path to MGM-Omni model"
    )
    parser.add_argument(
        "--cosyvoice_path", type=str, default=None, help="CosyVoice model path"
    )
    parser.add_argument("--lang", type=str, default="zh", help="Language")
    parser.add_argument(
        "--load_8bit", action="store_true", help="Load model in 8-bit mode"
    )
    parser.add_argument(
        "--load_4bit", action="store_true", help="Load model in 4-bit mode"
    )
    parser.add_argument(
        "--retry_times", type=int, default=5, help="Retry times for TTS generation"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    logger.info(f"Loading MGM-Omni model from {args.path}")

    # Model initialization
    disable_torch_init()

    # Load pretrained model
    tokenizer, model = load_pretrained_model(
        args.path,
        args.load_8bit,
        args.load_4bit,
        device=device,
        cosyvoice_path=args.cosyvoice_path,
    )

    logger.info("MGM-Omni model successfully loaded")

    # Determine prompt based on language
    pre_prompt_cn = "使用提供的音频片段的语气回复。"
    pre_prompt_en = "Respond with the tone of the reference audio clip."

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

            # Get reference audio path
            if args.lang == "zh":
                default_ref_audio = "third_party/MGM-Omni/assets/ref_audio/Man_ZH.wav"
                default_ref_text = (
                    "他疯狂寻找到能够让自己升级的办法终于有所收获，那就是炼体。"
                )
            else:
                default_ref_audio = "third_party/MGM-Omni/assets/ref_audio/Man_EN.wav"
                default_ref_text = '"Incredible!" Dr. Chen exclaimed, unable to contain her enthusiasm. "The quantum fluctuations we have observed in these superconducting materials exhibit completely unexpected characteristics."'

            if "prompt_audio" not in x:
                x["prompt_audio"] = default_ref_audio
                x["prompt_text"] = default_ref_text

            ref_audio = x["prompt_audio"]
            audio_refer, _ = librosa.load(ref_audio, sr=16000)
            audio_refer = torch.tensor(audio_refer).unsqueeze(0).to(model.device)

            # Get reference text
            if x.get("prompt_text", ""):
                text_refer = x.get("prompt_text", "")
            else:
                whispers_model = whisper.load_model("large-v3")
                text_refer = whispers_asr(whispers_model, ref_audio)

            input_ids_refer = tokenizer(text_refer)["input_ids"]
            input_ids_refer = (
                torch.tensor(input_ids_refer).unsqueeze(0).to(model.device)
            )

            # Detect language and choose prompt
            has_chinese = any("\u4e00" <= char <= "\u9fff" for char in text_refer)
            pre_prompt = pre_prompt_cn if has_chinese else pre_prompt_en

            # Prepare conversation
            conv = conv_templates["qwen2vl"].copy()
            roles = conv.roles

            oup = AUDIO_START + copy.deepcopy(x["text"])
            inp = pre_prompt + AUDIO_START + DEFAULT_SPEECH_TOKEN + AUDIO_END + "\n"

            conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], oup)
            prompt_text = conv.get_prompt()

            input_ids = (
                tokenizer_speech_token(prompt_text, tokenizer, return_tensors="pt")
                .unsqueeze(0)
                .to(model.device)
            )

            # Get sampling parameters
            temperature = x.get("temperature", 0.2)
            max_new_tokens = x.get("max_new_tokens", 8192)

            # TTS generation with retry mechanism
            with torch.inference_mode():
                current_temp = temperature
                tts_success = False
                logger.info(
                    f"TTS generation with temperature: {current_temp}, max_new_tokens: {max_new_tokens}, text_refer: {text_refer}"
                )
                for i in range(args.retry_times):
                    speech_ids, audio, tts_error = model.generate(
                        input_ids.clone(),
                        input_ids_refer=input_ids_refer.clone(),
                        audio_refer=audio_refer,
                        do_sample=True if current_temp > 0 else False,
                        temperature=current_temp,
                        max_new_tokens=max_new_tokens,
                        bos_token_id=tokenizer.pad_token_id,
                        eos_token_id=[tokenizer.eos_token_id],
                        pad_token_id=tokenizer.pad_token_id,
                        tokenizer=tokenizer,
                        check_tts_result=True,
                        use_cache=True,
                    )

                    if not tts_error:
                        if i > 0:
                            logger.info(
                                f"Trial {i} Success, speech_ids shape: {speech_ids.shape}"
                            )
                        tts_success = True
                        break

                    logger.warning(
                        f"Trial {i} Failed, speech_ids shape: {speech_ids.shape}"
                    )
                    current_temp = max(1.0, current_temp + 0.1)

                if not tts_success:
                    logger.error(
                        f"TTS generation failed after {args.retry_times} retries"
                    )
                    print(
                        f"Error: TTS generation failed after {args.retry_times} retries",
                        flush=True,
                    )
                    continue

            # Convert to numpy
            audio = to_numpy(audio)

            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                sf.write(f.name, audio, samplerate=24000)

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
            logger.error(f"Error processing request: {str(e)}")
            print(f"Error: {str(e)}", flush=True)
