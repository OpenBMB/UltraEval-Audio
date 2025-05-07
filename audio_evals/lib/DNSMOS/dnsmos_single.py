import argparse
import json
import os

import librosa
import numpy as np
import onnxruntime as ort
import soundfile as sf

SAMPLING_RATE = 16000
INPUT_LENGTH = 9.01


class ComputeScore:
    def __init__(self, model_path, p_model_path, p808_model_path) -> None:
        self.onnx_sess = ort.InferenceSession(model_path)
        self.p_onnx_sess = ort.InferenceSession(p_model_path)
        self.p808_onnx_sess = ort.InferenceSession(p808_model_path)

    def audio_melspec(
        self, audio, n_mels=120, frame_size=320, hop_length=160, sr=16000, to_db=True
    ):
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_fft=frame_size + 1, hop_length=hop_length, n_mels=n_mels
        )
        if to_db:
            mel_spec = (librosa.power_to_db(mel_spec, ref=np.max) + 40) / 40
        return mel_spec.T

    def get_polyfit_val(self, sig, bak, ovr, is_personalized_MOS):
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

    def __call__(self, fpath, sampling_rate):
        if not fpath.endswith(".wav"):
            raise ValueError("must be wav file")
        aud, input_fs = sf.read(fpath)
        fs = sampling_rate
        if input_fs != fs:
            audio = librosa.resample(aud, orig_sr=input_fs, target_sr=fs)
        else:
            audio = aud
        actual_audio_len = len(audio)
        len_samples = int(INPUT_LENGTH * fs)
        while len(audio) < len_samples:
            audio = np.append(audio, audio)

        num_hops = int(np.floor(len(audio) / fs) - INPUT_LENGTH) + 1
        hop_len_samples = fs
        predicted_mos_sig_seg_raw = []
        predicted_mos_bak_seg_raw = []
        predicted_mos_ovr_seg_raw = []
        predicted_mos_sig_seg = []
        predicted_mos_bak_seg = []
        predicted_mos_ovr_seg = []
        predicted_p808_mos = []
        predicted_p_mos_sig_seg = []
        predicted_p_mos_bak_seg = []
        predicted_p_mos_ovr_seg = []

        for idx in range(num_hops):
            audio_seg = audio[
                int(idx * hop_len_samples) : int((idx + INPUT_LENGTH) * hop_len_samples)
            ]
            if len(audio_seg) < len_samples:
                continue

            input_features = np.array(audio_seg).astype("float32")[np.newaxis, :]
            p808_input_features = np.array(
                self.audio_melspec(audio=audio_seg[:-160])
            ).astype("float32")[np.newaxis, :, :]
            oi = {"input_1": input_features}
            p808_oi = {"input_1": p808_input_features}
            p808_mos = self.p808_onnx_sess.run(None, p808_oi)[0][0][0]
            mos_sig_raw, mos_bak_raw, mos_ovr_raw = self.onnx_sess.run(None, oi)[0][0]
            mos_sig, mos_bak, mos_ovr = self.get_polyfit_val(
                mos_sig_raw, mos_bak_raw, mos_ovr_raw, False
            )
            predicted_mos_sig_seg_raw.append(mos_sig_raw)
            predicted_mos_bak_seg_raw.append(mos_bak_raw)
            predicted_mos_ovr_seg_raw.append(mos_ovr_raw)
            predicted_mos_sig_seg.append(mos_sig)
            predicted_mos_bak_seg.append(mos_bak)
            predicted_mos_ovr_seg.append(mos_ovr)
            predicted_p808_mos.append(p808_mos)

            mos_sig_raw, mos_bak_raw, mos_ovr_raw = self.p_onnx_sess.run(None, oi)[0][0]
            mos_sig, mos_bak, mos_ovr = self.get_polyfit_val(
                mos_sig_raw, mos_bak_raw, mos_ovr_raw, True
            )
            predicted_p_mos_sig_seg.append(mos_sig)
            predicted_p_mos_bak_seg.append(mos_bak)
            predicted_p_mos_ovr_seg.append(mos_ovr)

        clip_dict = {"filename": fpath, "len_in_sec": actual_audio_len / fs, "sr": fs}
        clip_dict["num_hops"] = num_hops
        clip_dict["OVRL_raw"] = float(np.mean(predicted_mos_ovr_seg_raw))
        clip_dict["SIG_raw"] = float(np.mean(predicted_mos_sig_seg_raw))
        clip_dict["BAK_raw"] = float(np.mean(predicted_mos_bak_seg_raw))
        clip_dict["OVRL"] = float(np.mean(predicted_mos_ovr_seg))
        clip_dict["SIG"] = float(np.mean(predicted_mos_sig_seg))
        clip_dict["BAK"] = float(np.mean(predicted_mos_bak_seg))
        clip_dict["P808_MOS"] = float(np.mean(predicted_p808_mos))
        clip_dict["P_SIG"] = float(np.mean(predicted_p_mos_sig_seg))
        clip_dict["P_BAK"] = float(np.mean(predicted_p_mos_bak_seg))
        clip_dict["P_OVRL"] = float(np.mean(predicted_p_mos_ovr_seg))
        return clip_dict


p808_model_path = os.path.join("DNSMOS", "model_v8.onnx")
model_path = os.path.join("DNSMOS", "sig_bak_ovr.onnx")
primary_model_path = os.path.join("pDNSMOS", "sig_bak_ovr.onnx")
compute_score = ComputeScore(model_path, primary_model_path, p808_model_path)


def main(args):
    res = compute_score(args.file_name, SAMPLING_RATE)
    with open(args.output_file, "w") as f:
        json.dump(res, f, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f", "--file-name", required=True, help="audio clips in .wav to be evaluated"
    )
    parser.add_argument(
        "-o", "--output-file", required=True, help="Output file name for the results"
    )

    args = parser.parse_args()

    main(args)
