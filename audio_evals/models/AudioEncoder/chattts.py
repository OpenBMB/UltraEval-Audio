import logging
import os
import tempfile
from dataclasses import asdict
import torch
import soundfile as sf
import librosa
from typing import Dict
from vocos import Vocos
from vocos.pretrained import instantiate_class

from audio_evals.base import PromptStruct
from audio_evals.lib.chattts import VocosConfig, DVAEConfig, DVAE
from audio_evals.models.model import Model


logger = logging.getLogger(__name__)


class ChatTTSModel(Model):
    def __init__(self, model_path: str, sample_params: Dict[str, any] = None):
        super().__init__(True, sample_params)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        vocos_ckpt_path = os.path.join(model_path, "Vocos.pt")
        dvae_ckpt_path = os.path.join(model_path, "DVAE_full.pt")

        vocos_config = VocosConfig()
        feature_extractor = instantiate_class(
            args=(), init=asdict(vocos_config.feature_extractor)
        )
        backbone = instantiate_class(args=(), init=asdict(vocos_config.backbone))
        head = instantiate_class(args=(), init=asdict(vocos_config.head))
        vocos = (
            Vocos(feature_extractor=feature_extractor, backbone=backbone, head=head)
            .to(self.device)
            .eval()
        )
        vocos.load_state_dict(torch.load(vocos_ckpt_path))
        self.vocos = vocos

        dvae_config = DVAEConfig()
        dvae = DVAE(
            decoder_config=asdict(dvae_config.decoder),
            encoder_config=asdict(dvae_config.encoder),
            vq_config=asdict(dvae_config.vq),
            dim=dvae_config.decoder.idim,
            coef=None,
            device=self.device,
        )
        dvae.load_pretrained(dvae_ckpt_path, self.device)

        self.dvae = dvae.eval()
        logger.info("model loaded successfully")

    def _inference(self, prompt: PromptStruct, **kwargs) -> str:
        audio_path = prompt["audio"]
        logger.debug(f"Processing audio file: {audio_path}")

        y, sr = librosa.load(audio_path, sr=24000, mono=True)
        waveform = torch.tensor(y).to(self.device)
        x = self.dvae(waveform, "encode")
        reconstructed_mel = self.dvae(x, "decode")
        reconstructed_waveform = self.vocos.decode(reconstructed_mel).cpu().numpy()

        waveform_mono = reconstructed_waveform.squeeze()
        # 保存生成的音频到临时文件
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, waveform_mono, samplerate=24000, subtype="PCM_16")
            return f.name
