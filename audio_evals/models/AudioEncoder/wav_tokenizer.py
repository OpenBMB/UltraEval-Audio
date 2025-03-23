import sys

sys.path.append("audio_evals/lib/WavTokenizer")
import logging
import os.path
import tempfile


from typing import Dict
from audio_evals.lib.WavTokenizer.encoder.utils import convert_audio
import torchaudio
import torch
from audio_evals.lib.WavTokenizer.decoder.pretrained import WavTokenizer
from audio_evals.base import PromptStruct
from audio_evals.models.model import OfflineModel

logger = logging.getLogger(__name__)


class WavTokenizerEncoder(OfflineModel):

    def __init__(
        self,
        config_name: str,
        model_path: str,
        sample_params: Dict[str, any] = None,
    ):
        super().__init__(is_chat=True, sample_params=sample_params)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.config_name = os.path.join(
            "audio_evals/lib/WavTokenizer/configs", config_name
        )

        logger.info(f"Loading WavTokenizer from {model_path}")
        self.model = WavTokenizer.from_pretrained0802(self.config_name, model_path)
        self.model = self.model.to(self.device)
        logger.info(f"WavTokenizer loaded on {self.device}")

    def _inference(self, prompt: PromptStruct, **kwargs) -> str:
        audio_path = prompt["audio"]

        wav, sr = torchaudio.load(audio_path)
        wav = convert_audio(wav, sr, 24000, 1)
        bandwidth_id = torch.tensor([0])
        wav = wav.to(self.device)
        features, discrete_code = self.model.encode_infer(
            wav, bandwidth_id=bandwidth_id
        )
        audio_out = self.model.decode(
            features, bandwidth_id=bandwidth_id.to(self.device)
        )

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            torchaudio.save(
                f.name,
                audio_out.cpu(),
                sample_rate=24000,
                encoding="PCM_S",
                bits_per_sample=16,
            )
            return f.name
