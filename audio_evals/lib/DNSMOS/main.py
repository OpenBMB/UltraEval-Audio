import argparse
import json
import os
import select
import sys
import time
import traceback

import librosa
import numpy as np
import onnxruntime as ort
import soundfile as sf

SAMPLING_RATE = 16000
INPUT_LENGTH = 9.01


class ComputeScore:
    def __init__(self, model_path, p_model_path, p808_model_path) -> None:
        """Initializes the ComputeScore class with ONNX model paths."""
        try:
            self.onnx_sess = ort.InferenceSession(model_path)
            self.p_onnx_sess = ort.InferenceSession(p_model_path)
            self.p808_onnx_sess = ort.InferenceSession(p808_model_path)
            print("DNSMOS models loaded successfully.", flush=True)
        except Exception as e:
            print(f"Error loading ONNX models: {e}", flush=True)
            raise

    def audio_melspec(
        self, audio, n_mels=120, frame_size=320, hop_length=160, sr=16000, to_db=True
    ):
        """Calculates the Mel spectrogram of an audio signal."""
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_fft=frame_size + 1, hop_length=hop_length, n_mels=n_mels
        )
        if to_db:
            mel_spec = (librosa.power_to_db(mel_spec, ref=np.max) + 40) / 40
        return mel_spec.T

    def get_polyfit_val(self, sig, bak, ovr, is_personalized_MOS):
        """Applies polynomial fitting to raw scores."""
        if is_personalized_MOS:
            p_ovr = np.poly1d([-0.00533021, 0.005101, 1.18058466, -0.11236046])
            p_sig = np.poly1d([-0.01019296, 0.02751166, 1.19576786, -0.24348726])
            p_bak = np.poly1d([-0.04976499, 0.44276479, -0.1644611, 0.96883132])
        else:
            p_ovr = np.poly1d([-0.06766283, 1.11546468, 0.04602535])
            p_sig = np.poly1d([-0.08397278, 1.22083953, 0.0052439])
            p_bak = np.poly1d([-0.13166888, 1.60915514, -0.39604546])

        sig_poly = p_sig(sig)
        bak_poly = p_bak(bak)
        ovr_poly = p_ovr(ovr)

        return sig_poly, bak_poly, ovr_poly

    def __call__(self, fpath, sampling_rate=SAMPLING_RATE):
        """Computes DNSMOS scores for a given audio file path."""
        if not os.path.exists(fpath):
            raise ValueError(f"Audio file not found: {fpath}")
        # Allow various audio formats, not just wav
        # if not fpath.endswith('.wav'):
        #     raise ValueError('Input must be a .wav file')

        try:
            aud, input_fs = sf.read(fpath)
        except Exception as e:
            raise ValueError(f"Error reading audio file {fpath}: {e}")

        fs = sampling_rate
        if input_fs != fs:
            # Handle potential multi-channel audio by taking the mean
            if aud.ndim > 1:
                aud = np.mean(aud, axis=1)
            audio = librosa.resample(aud, orig_sr=input_fs, target_sr=fs)
        else:
            # Handle potential multi-channel audio
            if aud.ndim > 1:
                audio = np.mean(aud, axis=1)
            else:
                audio = aud

        actual_audio_len = len(audio)
        len_samples = int(INPUT_LENGTH * fs)

        # Pad audio if it's shorter than INPUT_LENGTH
        if len(audio) < len_samples:
            audio = np.pad(audio, (0, len_samples - len(audio)), "constant")
        # else: # If longer, process in segments
        #     # This part seems complex and might need adjustment based on how segmentation should work
        #     # For now, let's process only the first segment if longer, or the whole if shorter/equal
        #     audio = audio[:len_samples] # Simplified: process only the beginning

        # Recalculate based on potentially padded/truncated audio
        current_audio_len = len(audio)

        # If audio is shorter than the required input length after padding, something is wrong
        if current_audio_len < len_samples:
            raise ValueError(
                f"Audio length ({current_audio_len}) is less than required input length ({len_samples}) even after padding."
            )

        # --- Simplified processing: Use only the first segment ---
        audio_seg = audio[:len_samples]

        input_features = np.array(audio_seg).astype("float32")[np.newaxis, :]
        # Ensure the mel spec input length matches model expectations (adjust slicing if needed)
        # Original dnsmos_single used audio_seg[:-160], check if this is necessary for model_v8
        mel_input_audio = audio_seg  # Use the full segment for mel spec unless model requires specific length
        if len(mel_input_audio) < 160:  # Prevent error if audio is extremely short
            mel_input_audio = np.pad(
                mel_input_audio, (0, 160 - len(mel_input_audio)), "constant"
            )

        p808_input_features = np.array(
            self.audio_melspec(audio=mel_input_audio[:-160])
        ).astype("float32")[np.newaxis, :, :]

        oi = {"input_1": input_features}
        p808_oi = {"input_1": p808_input_features}

        # Run inference
        mos_sig_raw, mos_bak_raw, mos_ovr_raw = self.onnx_sess.run(None, oi)[0][0]
        p_mos_sig_raw, p_mos_bak_raw, p_mos_ovr_raw = self.p_onnx_sess.run(None, oi)[0][
            0
        ]
        p808_mos = self.p808_onnx_sess.run(None, p808_oi)[0][0][0]

        # Apply polyfit
        mos_sig, mos_bak, mos_ovr = self.get_polyfit_val(
            mos_sig_raw, mos_bak_raw, mos_ovr_raw, False
        )
        p_mos_sig, p_mos_bak, p_mos_ovr = self.get_polyfit_val(
            p_mos_sig_raw, p_mos_bak_raw, p_mos_ovr_raw, True
        )

        # --- Simplified result dictionary (no segmentation means) ---
        clip_dict = {
            "OVRL": float(mos_ovr),
            "SIG": float(mos_sig),
            "BAK": float(mos_bak),
            "P808_MOS": float(p808_mos),
            "P_SIG": float(p_mos_sig),
            "P_BAK": float(p_mos_bak),
            "P_OVRL": float(p_mos_ovr),
        }

        return clip_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to sig_bak_ovr.onnx"
    )
    parser.add_argument(
        "--p_model_path",
        type=str,
        required=True,
        help="Path to personalized sig_bak_ovr.onnx",
    )
    parser.add_argument(
        "--p808_model_path",
        type=str,
        required=True,
        help="Path to model_v8.onnx (P.808)",
    )
    args = parser.parse_args()

    try:
        compute_score = ComputeScore(
            args.model_path, args.p_model_path, args.p808_model_path
        )
    except Exception as e:
        print(f"Failed to initialize ComputeScore: {e}", file=sys.stderr, flush=True)
        sys.exit(1)

    print("DNSMOS main.py started. Waiting for input...", flush=True)

    while True:
        try:
            prompt = input()
            anchor = prompt.find("->")
            if anchor == -1:
                print(
                    f"Error: Invalid input format. Expected 'prefix->filepath', got '{prompt}'",
                    file=sys.stderr,
                    flush=True,
                )
                continue

            prefix = prompt[:anchor].strip() + "->"
            filepath = prompt[anchor + 2 :].strip()

            print(f"Received request: {filepath}", flush=True)  # Debug print
            results = compute_score(filepath)
            result_json = json.dumps(results, ensure_ascii=False)

            # Wait for acknowledgment from the client
            ack_received = False
            ack_wait_start = time.time()
            while time.time() - ack_wait_start < 60:  # 60s timeout for ack
                print(f"{prefix}{result_json}", flush=True)
                print(
                    f"Sent results for: {filepath}. Waiting for ack...", flush=True
                )  # Debug print
                rlist, _, xlist = select.select([sys.stdin], [], [sys.stdin], 1.0)
                if xlist:
                    print(
                        f"Error: stdin broken while waiting for ack for {prefix}",
                        file=sys.stderr,
                        flush=True,
                    )
                    break  # Treat as error, break inner loop
                if rlist:
                    ack_signal = sys.stdin.readline().strip()
                    expected_ack = (
                        f"{prefix.strip('->')}->ok"  # Construct expected ack signal
                    )
                    if ack_signal == expected_ack:
                        print(f"Received ack for {prefix}", flush=True)
                        ack_received = True
                        break  # Ack received, break inner loop
                    else:
                        print(
                            f"Warning: Received unexpected input while waiting for ack for {prefix}: '{ack_signal}'. Potential de-sync? Expected '{expected_ack}'",
                            file=sys.stderr,
                            flush=True,
                        )

            if not ack_received:
                print(
                    f"Warning: Did not receive ack for {prefix} within timeout.",
                    file=sys.stderr,
                    flush=True,
                )
                # Continue to the next request anyway, but log the warning.

        except EOFError:
            print("Error:Input stream closed. Exiting.", flush=True)
            break  # Exit loop if stdin is closed
        except Exception as e:
            traceback.extract_stack()
            print(f"Error: in main loop: {e}", file=sys.stderr, flush=True)
