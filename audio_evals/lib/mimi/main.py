import argparse
import logging
import tempfile

import numpy as np
import torch
import librosa
import soundfile as sf

from transformers import MimiModel, AutoFeatureExtractor

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", type=str, required=True, help="Path to checkpoint file"
    )
    parser.add_argument("--mono", type=bool, required=False, help="must be momo sample")
    parser.add_argument(
        "--stereo", type=bool, required=False, help="must be stereo sample"
    )
    config = parser.parse_args()

    model_path = config.path
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading Mimi model from {model_path}")
    processor = AutoFeatureExtractor.from_pretrained(model_path)
    model = MimiModel.from_pretrained(model_path).to(device)
    model.eval()
    logger.info(f"Mimi model loaded on {device}")

    while True:
        try:
            audio = input()
            input_array = librosa.load(audio, sr=processor.sampling_rate)[0]
            if config.mono:
                input_array = librosa.load(
                    audio, sr=processor.sampling_rate, mono=True
                )[0]
            elif config.stereo and input_array.ndim == 1:
                input_array = np.stack([input_array, input_array], axis=0)

            inputs = processor(
                raw_audio=input_array,
                sampling_rate=processor.sampling_rate,
                return_tensors="pt",
            )
            for k, v in inputs.items():
                inputs[k] = v.to(device)

            encoder_outputs = model.encode(inputs["input_values"])
            audio_values = model.decode(
                encoder_outputs.audio_codes,
            )[0]
            audio_values = audio_values.squeeze()
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                audio_values = audio_values.cpu().detach().numpy()
                if audio_values.ndim == 2:
                    audio_values = audio_values.T
                sf.write(f.name, audio_values, processor.sampling_rate)
                print("Result:{}".format(f.name))
        except Exception as e:
            print("Error:{}".format(e))
