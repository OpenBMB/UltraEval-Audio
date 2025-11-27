import argparse
import json
import logging
import select
import sys
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import soundfile as sf
import scipy


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True, help="Path to Whisper model")
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=0,
        help="Chunk size in seconds (0 means no chunking)",
    )
    config = parser.parse_args()

    # Initialize model
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = WhisperForConditionalGeneration.from_pretrained(config.path)
    model.to(device)
    processor = WhisperProcessor.from_pretrained(config.path)
    logger.info(f"Using Whisper model from: {config.path} on device: {device}")
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
            logger.info(f"Received input: {x}")
            wav, sr = sf.read(x["audio"])
            if sr != 16000:
                wav = scipy.signal.resample(wav, int(len(wav) * 16000 / sr))
                sr = 16000

            if config.chunk_size > 0:
                # Use chunking
                chunk_size = config.chunk_size * sr
                texts = []
                for start in range(0, len(wav), chunk_size):
                    chunk = wav[start : start + chunk_size]
                    input_features = processor(
                        chunk, sampling_rate=16000, return_tensors="pt"
                    ).input_features
                    input_features = input_features.to(device)
                    forced_decoder_ids = processor.get_decoder_prompt_ids(
                        language=x.get("language", "english"), task="transcribe"
                    )
                    with torch.no_grad():
                        predicted_ids = model.generate(
                            input_features, forced_decoder_ids=forced_decoder_ids
                        )
                    text = processor.batch_decode(
                        predicted_ids, skip_special_tokens=True
                    )[0]
                    texts.append(text.strip())
                transcription = " ".join(t for t in texts if t)
            else:
                # Process entire audio at once
                input_features = processor(
                    wav, sampling_rate=16000, return_tensors="pt"
                ).input_features
                input_features = input_features.to(device)
                forced_decoder_ids = processor.get_decoder_prompt_ids(
                    language=x.get("language", "english"), task="transcribe"
                )
                predicted_ids = model.generate(
                    input_features, forced_decoder_ids=forced_decoder_ids
                )
                transcription = processor.batch_decode(
                    predicted_ids, skip_special_tokens=True
                )[0]
            result = {"text": transcription}
            retry = 3
            while retry:
                print(f"{prefix}{result['text']}", flush=True)
                rlist, _, _ = select.select([sys.stdin], [], [], 1)
                if rlist:
                    finish = sys.stdin.readline().strip()
                    if finish == "{}close".format(prefix):
                        break
                print("not found close signal, will emit again", flush=True)
                retry -= 1
        except Exception as e:
            print(f"Error: {str(e)}", flush=True)
