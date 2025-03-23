import logging
import tempfile

import torch
import soundfile as sf
import librosa
from typing import Dict
from vocos import Vocos
from vocos.pretrained import instantiate_class

from audio_evals.base import PromptStruct
from audio_evals.models.model import Model

logger = logging.getLogger(__name__)


class VocosModel(Model):
    def __init__(
        self,
        model_path: str,
        feature_extractor: Dict[str, any],
        backbone: Dict[str, any],
        head: Dict[str, any],
        sample_params: Dict[str, any] = None,
    ):
        super().__init__(True, sample_params)  # 作为非聊天模型
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading Vocos model from {model_path} to device {self.device}")
        feature_extractor = instantiate_class(args=(), init=feature_extractor)
        backbone = instantiate_class(args=(), init=backbone)
        head = instantiate_class(args=(), init=head)
        self.model = Vocos(
            feature_extractor=feature_extractor, backbone=backbone, head=head
        )
        self.model.to(self.device)
        self.model.eval()
        self.model.load_state_dict(torch.load(model_path, weights_only=True, mmap=True))
        logger.info("Vocos model loaded successfully")

    def _inference(self, prompt: PromptStruct, **kwargs) -> str:
        audio_path = prompt["audio"]
        logger.debug(f"Processing audio file: {audio_path}")
        y, sr = librosa.load(audio_path, sr=None)
        waveform = torch.tensor(y).unsqueeze(0).to(self.device)
        generated_audio = self.model(waveform)

        # 保存生成的音频到临时文件
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, generated_audio.squeeze().cpu().numpy(), samplerate=22050)
            return f.name
