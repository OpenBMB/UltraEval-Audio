import logging
import re
import tempfile
from typing import Dict

import numpy as np
import soundfile as sf
import librosa
import torch
from transformers import EncodecModel, AutoProcessor

from audio_evals.base import PromptStruct
from audio_evals.models.model import Model

logger = logging.getLogger(__name__)


class Encodec(Model):
    def __init__(
        self,
        path: str,
        mono: bool = False,
        stereo: bool = False,
        sample_params: Dict[str, any] = None,
    ):
        super().__init__(True, sample_params)  # as a chat model
        # Load the speech recognition model
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        logger.info("start load model from {} to device {}".format(path, self.device))
        self.model = EncodecModel.from_pretrained(path)
        self.model.to(self.device)
        logger.info("successfully load model from {}".format(path))
        self.mono = mono
        self.stereo = stereo
        # Load the processor
        self.processor = AutoProcessor.from_pretrained(path)

    def _inference(self, prompt: PromptStruct, **kwargs) -> str:
        audio = prompt["audio"]
        logger.debug(f"the input is {audio}, {kwargs}")

        input_array = librosa.load(audio, sr=self.processor.sampling_rate)[0]
        if self.mono:
            input_array = librosa.load(
                audio, sr=self.processor.sampling_rate, mono=True
            )[0]
        elif self.stereo and input_array.ndim == 1:
            input_array = np.stack([input_array, input_array], axis=0)

        inputs = self.processor(
            raw_audio=input_array,
            sampling_rate=self.processor.sampling_rate,
            return_tensors="pt",
        )
        for k, v in inputs.items():
            inputs[k] = v.to(self.device)
        # Generate the audio
        # explicitly encode then decode the audio inputs
        encoder_outputs = self.model.encode(
            inputs["input_values"], inputs["padding_mask"]
        )
        audio_values = self.model.decode(
            encoder_outputs.audio_codes,
            encoder_outputs.audio_scales,
            inputs["padding_mask"],
        )[0]
        audio_values = audio_values.squeeze()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            audio_values = audio_values.cpu().detach().numpy()
            if audio_values.ndim == 2:
                audio_values = audio_values.T
            sf.write(f.name, audio_values, self.processor.sampling_rate)
            return f.name
