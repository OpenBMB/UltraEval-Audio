import argparse
import select
import sys
import tempfile
import os
import torch
import soundfile as sf
import librosa
import logging
from dataclasses import asdict
from vocos import Vocos
from vocos.pretrained import instantiate_class
from audio_evals.lib.ChatTTS.chattts import VocosConfig, DVAEConfig, DVAE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model checkpoint directory",
    )
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    model_path = args.model_path
    vocos_ckpt_path = os.path.join(model_path, "asset", "Vocos.pt")
    dvae_ckpt_path = os.path.join(model_path, "asset", "DVAE_full.pt")

    vocos_config = VocosConfig()
    feature_extractor = instantiate_class(
        args=(), init=asdict(vocos_config.feature_extractor)
    )
    backbone = instantiate_class(args=(), init=asdict(vocos_config.backbone))
    head = instantiate_class(args=(), init=asdict(vocos_config.head))
    vocos = (
        Vocos(feature_extractor=feature_extractor, backbone=backbone, head=head)
        .to(device)
        .eval()
    )
    vocos.load_state_dict(torch.load(vocos_ckpt_path))

    dvae_config = DVAEConfig()
    dvae = DVAE(
        decoder_config=asdict(dvae_config.decoder),
        encoder_config=asdict(dvae_config.encoder),
        vq_config=asdict(dvae_config.vq),
        dim=dvae_config.decoder.idim,
        coef=None,
        device=device,
    )
    dvae.load_pretrained(dvae_ckpt_path, device)
    dvae = dvae.eval()

    print(f"Model loaded from checkpoint: {model_path}", flush=True)

    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break
            prompt = line.strip()
            if not prompt:
                continue

            # 解析 uuid 前缀
            anchor = prompt.find("->")
            if anchor == -1:
                print(
                    "Error: Invalid format, must contain ->, got: {}".format(prompt),
                    flush=True,
                )
                continue

            prefix = prompt[: anchor + 2]  # 包含 "->"
            audio_path = prompt[anchor + 2 :]

            # Inference logic
            y, sr = librosa.load(audio_path, sr=24000, mono=True)
            waveform = torch.tensor(y).to(device)
            x = dvae(waveform, "encode")
            reconstructed_mel = dvae(x, "decode")
            reconstructed_waveform = (
                vocos.decode(reconstructed_mel).cpu().detach().numpy()
            )

            waveform_mono = reconstructed_waveform.squeeze()

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                sf.write(f.name, waveform_mono, samplerate=24000, subtype="PCM_16")

                # 发送结果并等待 close 信号确认
                retry = 3
                while retry:
                    retry -= 1
                    print(f"{prefix}{f.name}", flush=True)
                    rlist, _, _ = select.select([sys.stdin], [], [], 1)
                    if rlist:
                        finish = sys.stdin.readline().strip()
                        if finish == f"{prefix}close":
                            break
                        logger.info(
                            f"not found close signal, got: {finish}, will emit again"
                        )
        except EOFError:
            break
        except Exception as e:
            print(f"Error:{e}", flush=True)
            logger.exception("Error during inference")
