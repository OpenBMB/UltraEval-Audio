import subprocess
import sys

sys.path.append("/DATA/disk1/home/shiqundong/project/CosyVoice/third_party/Matcha-TTS")
sys.path.append(
    "/DATA/disk1/home/shiqundong/project/CosyVoice/env/lib/python3.10/site-packages/"
)
sys.path.append("/DATA/disk1/home/shiqundong/project/CosyVoice")

import logging
import os
import tempfile
import torch
import soundfile as sf
import librosa
from typing import Dict
import s3tokenizer
from audio_evals.base import PromptStruct
from audio_evals.models.model import OfflineModel
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav

logger = logging.getLogger(__name__)


class CosyVoiceEncoder(OfflineModel):
    def __init__(self, model_path: str, sample_params: Dict[str, any] = None):
        super().__init__(True, sample_params)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.tokenizer = s3tokenizer.load_model("speech_tokenizer_v2_25hz")
        self.tokenizer.to(self.device)

        self.model = CosyVoice2(model_path, load_jit=False, load_trt=False, fp16=False)
        logger.info("model loaded successfully")

    def _inference(self, prompt: PromptStruct, **kwargs) -> str:
        audio_path = prompt["audio"]
        logger.debug(f"Processing audio file: {audio_path}")

        x = load_wav(audio_path, 16_000)
        mel = s3tokenizer.log_mel_spectrogram(x.squeeze(0))
        mels, mels_lens = s3tokenizer.padding([mel])
        audio_tokens = self.tokenizer.quantize(
            mels.to(self.device), mels_lens.to(self.device)
        )[0]

        waveform = x.to(self.device)
        sr = torch.tensor(self.model.sample_rate).to(self.device)
        model_input = self.model.frontend.frontend_token2wav(waveform, sr)
        wav_out = self.model.model.token2wav(
            token=audio_tokens,
            prompt_token=model_input["flow_prompt_speech_token"],
            prompt_feat=model_input["prompt_speech_feat"],
            embedding=model_input["flow_embedding"],
            uuid=None,
            token_offset=0,
            speed=1.0,
        )
        wav_out = wav_out.squeeze()
        # 保存生成的音频到临时文件
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, wav_out.cpu().numpy(), samplerate=self.model.sample_rate)
            return f.name
