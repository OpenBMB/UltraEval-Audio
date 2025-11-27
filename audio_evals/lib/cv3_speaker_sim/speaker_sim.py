# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

"""
Interactive speaker similarity evaluation script.
This script runs in interactive mode, accepting audio pairs and computing similarity scores.
"""

import os
import sys
import argparse
import select
import torch
import torchaudio
import numpy as np
import kaldiio

sys.path.append("audio_evals/lib/cv3_speaker_sim/3D-Speaker")
from speakerlab.process.processor import FBank

from speakerlab.utils.builder import dynamic_import

CAMPPLUS_VOX = {
    "obj": "speakerlab.models.campplus.DTDNN.CAMPPlus",
    "args": {
        "feat_dim": 80,
        "embedding_size": 512,
    },
}

CAMPPLUS_COMMON = {
    "obj": "speakerlab.models.campplus.DTDNN.CAMPPlus",
    "args": {
        "feat_dim": 80,
        "embedding_size": 192,
    },
}

ERes2Net_VOX = {
    "obj": "speakerlab.models.eres2net.ResNet.ERes2Net",
    "args": {
        "feat_dim": 80,
        "embedding_size": 192,
    },
}

ERes2Net_COMMON = {
    "obj": "speakerlab.models.eres2net.ResNet_aug.ERes2Net",
    "args": {
        "feat_dim": 80,
        "embedding_size": 192,
    },
}

ERes2Net_Base_3D_Speaker = {
    "obj": "speakerlab.models.eres2net.ResNet.ERes2Net",
    "args": {
        "feat_dim": 80,
        "embedding_size": 512,
        "m_channels": 32,
    },
}

ERes2Net_Large_3D_Speaker = {
    "obj": "speakerlab.models.eres2net.ResNet.ERes2Net",
    "args": {
        "feat_dim": 80,
        "embedding_size": 512,
        "m_channels": 64,
    },
}

supports = {
    "damo/speech_campplus_sv_en_voxceleb_16k": {
        "revision": "v1.0.2",
        "model": CAMPPLUS_VOX,
        "model_pt": "campplus_voxceleb.bin",
    },
    "damo/speech_campplus_sv_zh-cn_16k-common": {
        "revision": "v1.0.0",
        "model": CAMPPLUS_COMMON,
        "model_pt": "campplus_cn_common.bin",
    },
    "damo/speech_eres2net_sv_en_voxceleb_16k": {
        "revision": "v1.0.2",
        "model": ERes2Net_VOX,
        "model_pt": "pretrained_eres2net.ckpt",
    },
    "damo/speech_eres2net_sv_zh-cn_16k-common": {
        "revision": "v1.0.4",
        "model": ERes2Net_COMMON,
        "model_pt": "pretrained_eres2net_aug.ckpt",
    },
    "damo/speech_eres2net_base_sv_zh-cn_3dspeaker_16k": {
        "revision": "v1.0.1",
        "model": ERes2Net_Base_3D_Speaker,
        "model_pt": "eres2net_base_model.ckpt",
    },
    "damo/speech_eres2net_large_sv_zh-cn_3dspeaker_16k": {
        "revision": "v1.0.0",
        "model": ERes2Net_Large_3D_Speaker,
        "model_pt": "eres2net_large_model.ckpt",
    },
}


def load_wav(wav_file, obj_fs=16000):
    """Load audio file and resample to target sample rate."""
    if ".ark:" not in wav_file:
        wav, fs = torchaudio.load(wav_file)
        if wav.shape[0] > 1:
            wav = wav[0, :].unsqueeze(0)
    else:
        if " " in wav_file:
            fs = int(wav_file.split(" ")[1])
            wav_file = wav_file.split(" ")[0]
        else:
            fs = None
        retval = kaldiio.load_mat(wav_file)
        if isinstance(retval, tuple):
            if isinstance(retval[0], int):
                fs, wav = retval
            else:
                wav, fs = retval
        else:
            wav, fs = retval, fs if fs is not None else 16000

        if wav.dtype == np.int16:
            wav = wav / (2**16 - 1)
        elif wav.dtype == np.int32:
            wav = wav / (2**32 - 1)

        wav = torch.tensor(wav, dtype=torch.float32).unsqueeze(0)

    if fs != obj_fs:
        wav = torchaudio.functional.resample(wav, orig_freq=fs, new_freq=obj_fs)
        fs = obj_fs

    return wav


def compute_similarity(
    wav1_path, wav2_path, embedding_model, feature_extractor, device
):
    """Compute cosine similarity between two audio files."""
    try:
        # Load audio files
        wav1 = load_wav(wav1_path)
        wav2 = load_wav(wav2_path)

        # Compute features
        feat1 = feature_extractor(wav1).unsqueeze(0).to(device)
        feat2 = feature_extractor(wav2).unsqueeze(0).to(device)

        # Compute embeddings
        with torch.no_grad():
            emb1 = embedding_model(feat1).detach().cpu()
            emb2 = embedding_model(feat2).detach().cpu()

        # Compute similarity
        similarity = torch.nn.functional.cosine_similarity(emb1, emb2, dim=-1)
        return similarity.item()
    except Exception as e:
        raise RuntimeError(f"Failed to compute similarity: {str(e)}")


def main():
    parser = argparse.ArgumentParser(
        description="Interactive speaker similarity evaluation."
    )
    parser.add_argument(
        "--model_id",
        default="damo/speech_eres2net_base_sv_zh-cn_3dspeaker_16k",
        type=str,
        help="Model id",
    )
    parser.add_argument("--path", default="", type=str, help="Local model directory")
    parser.add_argument("--device", default="cuda:0", type=str, help="Device to use")

    args = parser.parse_args()

    # Validate model_id
    if args.model_id not in supports:
        print(f"Error: Model id '{args.model_id}' not currently supported.", flush=True)
        print(f"Supported models: {list(supports.keys())}", flush=True)
        sys.exit(1)

    conf = supports[args.model_id]
    pretrained_model = args.path

    if not os.path.exists(pretrained_model):
        print(f"Error: Model file not found at {pretrained_model}", flush=True)
        sys.exit(1)

    print(f"Loading model from {pretrained_model}", flush=True)
    pretrained_state = torch.load(pretrained_model, map_location="cpu")

    # Load model
    model = conf["model"]
    from speakerlab.models.eres2net.ERes2Net import ERes2Net as NET

    embedding_model = NET(**model["args"])
    embedding_model.load_state_dict(pretrained_state)
    embedding_model.eval()

    # Set device
    device = args.device if torch.cuda.is_available() else "cpu"
    embedding_model = embedding_model.to(device)

    # Initialize feature extractor
    feature_extractor = FBank(80, sample_rate=16000, mean_nor=True)

    print(f"Successfully loaded model on {device}", flush=True)
    print(
        "Ready to process audio pairs. Format: <prefix>-><audio1_path>,<audio2_path>",
        flush=True,
    )

    # Interactive loop
    while True:
        try:
            prompt = input()

            # Parse the input
            anchor = prompt.find("->")
            if anchor == -1:
                print(
                    f"Error: Invalid format, must contain  ->, but got: {prompt}",
                    flush=True,
                )
                continue

            prefix = prompt[:anchor].strip() + "->"
            wavs = prompt[anchor + 2 :].strip().split(",")

            if len(wavs) != 2:
                print(f"Error: Expected 2 audio files, got {len(wavs)}", flush=True)
                continue

            # Compute similarity
            try:
                similarity = compute_similarity(
                    wavs[0].strip(),
                    wavs[1].strip(),
                    embedding_model,
                    feature_extractor,
                    device,
                )

                # Send result with prefix
                retry = 3
                while retry:
                    print(f"{prefix}{similarity:.6f}", flush=True)
                    rlist, _, _ = select.select([sys.stdin], [], [], 1)
                    if rlist:
                        finish = sys.stdin.readline().strip()
                        if finish == f"{prefix}close":
                            break
                    print("not found close signal, will emit again", flush=True)
                    retry -= 1

            except Exception as e:
                print(f"Error: {str(e)}", flush=True)

        except EOFError:
            print("Received EOF, exiting...", flush=True)
            break
        except Exception as e:
            print(f"Error: {str(e)}", flush=True)


if __name__ == "__main__":
    main()
