import argparse
import logging
import tempfile
import os
import torch
import soundfile as sf
import torchaudio
import os
from transformers import MimiModel, AutoFeatureExtractor
import librosa


logger = logging.getLogger(__name__)


def load_wav(wav, target_sr):
    speech, sample_rate = torchaudio.load(wav)

    if sample_rate < target_sr or not wav.endswith("wav"):
        with tempfile.NamedTemporaryFile(suffix=".wav") as f:
            os.system(
                f"ffmpeg -i {wav} -y -ac 1 -ar {target_sr} -loglevel quiet {f.name}"
            )
            speech, sample_rate = torchaudio.load(f.name)

    if sample_rate != target_sr:
        assert (
            sample_rate > target_sr
        ), "wav sample rate {} must be greater than {}".format(sample_rate, target_sr)
        speech = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=target_sr
        )(speech)
    return speech


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
    print(f"Loading Mimi model from {model_path}")
    processor = AutoFeatureExtractor.from_pretrained(model_path)
    model = MimiModel.from_pretrained(model_path).to(device)
    model.eval()
    print(f"Mimi model loaded on {device}")
    sample_rate = 24000

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
            audio = prompt[anchor + 2 :].strip()
            if not audio:
                continue
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

            out_wav_chunks = []
            with torch.no_grad():
                encoder_outputs = model.encode(inputs["input_values"], num_quantizers=8)
                for i in range(encoder_outputs.audio_codes.shape[-1]):
                    out_wav_chunks.append(
                        model.decode(
                            encoder_outputs.audio_codes[:, :, i : i + 1]
                        ).audio_values
                    )

            audio_values = torch.cat(out_wav_chunks, dim=-1)
            audio_values = audio_values.squeeze()

            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    audio_values = audio_values.cpu().detach().numpy()
                    if audio_values.ndim == 2:
                        audio_values = audio_values.T
                    sf.write(f.name, audio_values, sample_rate)
                    print("Result:{}".format(f.name))
            finally:
                # Clean up GPU memory
                torch.cuda.empty_cache()
                # Reset model state
                model.zero_grad()
        except Exception as e:
            import traceback

            traceback.print_exc()
            print("Error:{}".format(e))
