import argparse
import json
import logging
import os
import select
import sys
import tempfile
import time
import traceback

import librosa
import numpy as np
import soundfile as sf
import torch
from transformers import EncodecModel, AutoProcessor

# Basic logging setup for the server script
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class EncodecProcessor:
    def __init__(self, model_path):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading Encodec model from {model_path} to device {self.device}")
        try:
            self.model = EncodecModel.from_pretrained(model_path)
            self.model.to(self.device)
            self.processor = AutoProcessor.from_pretrained(model_path)
            self.sampling_rate = self.processor.sampling_rate
            logger.info(
                f"Successfully loaded Encodec model and processor from {model_path}"
            )
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            raise

    def process_audio(self, audio_path, mono=False, stereo=False):
        """Encodes and decodes an audio file."""
        if not os.path.exists(audio_path):
            raise ValueError(f"Audio file not found: {audio_path}")

        try:
            input_array, sr = librosa.load(
                audio_path, sr=None
            )  # Load with original SR first
            if sr != self.sampling_rate:
                logger.warning(
                    f"Resampling audio from {sr} Hz to {self.sampling_rate} Hz"
                )
                input_array = librosa.resample(
                    input_array, orig_sr=sr, target_sr=self.sampling_rate
                )

            # Handle channel processing based on flags and input dimensions
            if mono:
                if input_array.ndim > 1:
                    input_array = librosa.to_mono(input_array)
                # If already mono, do nothing
            elif stereo:
                if input_array.ndim == 1:
                    logger.warning(
                        "Input is mono, duplicating channel for stereo output."
                    )
                    input_array = np.stack([input_array, input_array], axis=0)
                elif input_array.ndim == 2 and input_array.shape[0] != 2:
                    # If multi-channel but not stereo, convert to stereo (e.g., take first 2 channels or average)
                    logger.warning(
                        f"Input has {input_array.shape[0]} channels, converting to stereo."
                    )
                    input_array = input_array[:2, :]  # Example: take first two channels
                # If already stereo, do nothing

            # Ensure input_array matches expected dimensions for processor if needed
            # (Processor might handle mono/stereo automatically, check documentation)

            inputs = self.processor(
                raw_audio=input_array,
                sampling_rate=self.sampling_rate,
                return_tensors="pt",
            )

            # Move inputs to the correct device
            for k, v in inputs.items():
                inputs[k] = v.to(self.device)

            # Encode and decode
            with torch.no_grad():
                encoder_outputs = self.model.encode(
                    inputs["input_values"],
                    inputs.get("padding_mask"),  # Use get for optional padding_mask
                )
                audio_values = self.model.decode(
                    encoder_outputs.audio_codes,
                    encoder_outputs.audio_scales,
                    inputs.get("padding_mask"),  # Use get for optional padding_mask
                )[0]

            # Process output audio
            audio_values = audio_values.squeeze().cpu().detach().numpy()
            # Transpose if stereo (processor might output channels first)
            if audio_values.ndim == 2 and audio_values.shape[0] == 2:
                audio_values = (
                    audio_values.T
                )  # Shape (channels, samples) -> (samples, channels) for sf.write

            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                output_path = f.name
                sf.write(output_path, audio_values, self.sampling_rate)
            logger.info(f"Encoded/decoded audio saved to temporary file: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Error processing audio file {audio_path}: {e}")
            traceback.print_exc(file=sys.stderr)
            raise  # Re-raise the exception to be caught in the main loop


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", type=str, required=True, help="Path to the pretrained Encodec model"
    )
    args = parser.parse_args()

    try:
        processor = EncodecProcessor(args.path)
    except Exception as e:
        print(
            f"Failed to initialize EncodecProcessor: {e}", file=sys.stderr, flush=True
        )
        sys.exit(1)

    print("Encodec main.py server started. Waiting for input...", flush=True)

    while True:
        try:
            prompt = input()
            anchor = prompt.find("->")
            if anchor == -1:
                print(
                    f"Error: Invalid input format. Expected 'prefix->json_input', got '{prompt}'",
                    file=sys.stderr,
                    flush=True,
                )
                continue

            prefix = prompt[:anchor].strip() + "->"
            input_json_str = prompt[anchor + 2 :].strip()

            input_data = json.loads(input_json_str)
            audio_filepath = input_data.get("audio")
            mono_flag = input_data.get("mono", True)
            stereo_flag = input_data.get("stereo", False)

            if not audio_filepath:
                raise ValueError("Missing 'audio' key in input JSON.")

            print(
                f"Received request for: {audio_filepath} (mono={mono_flag}, stereo={stereo_flag})",
                flush=True,
            )

            output_filepath = processor.process_audio(
                audio_filepath, mono=mono_flag, stereo=stereo_flag
            )

            # Wait for acknowledgment
            ack_received = False
            ack_wait_start = time.time()
            while time.time() - ack_wait_start < 60:  # 60s timeout
                print(f"{prefix}{output_filepath}", flush=True)
                print(
                    f"Sent results for: {audio_filepath}. Waiting for ack...",
                    flush=True,
                )
                rlist, _, xlist = select.select([sys.stdin], [], [sys.stdin], 1.0)
                if xlist:
                    print(
                        f"Error: stdin broken while waiting for ack for {prefix}",
                        file=sys.stderr,
                        flush=True,
                    )
                    break
                if rlist:
                    ack_signal = sys.stdin.readline().strip()
                    expected_ack = f"{prefix.strip('->')}->ok"
                    if ack_signal == expected_ack:
                        break
                    else:
                        print(
                            f"Warning: Received unexpected input while waiting for ack for {prefix}: '{ack_signal}'. Expected '{expected_ack}'",
                            file=sys.stderr,
                            flush=True,
                        )
        except Exception as e:
            traceback.print_exc(file=sys.stderr)
            print(f"Error: in main loop: {e}", file=sys.stderr, flush=True)
