import logging
import re
from typing import Dict

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from audio_evals.base import PromptStruct
from audio_evals.models.model import Model

logger = logging.getLogger(__name__)


class WhisperModel(Model):
    def __init__(
        self,
        path: str,
        sample_params: Dict[str, any] = None,
    ):
        super().__init__(True, sample_params)  # as a chat model
        # Load the speech recognition model
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        logger.info("start load model from {} to device {}".format(path, self.device))

        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            path,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        ).eval()
        self.model.to(self.device)

        logger.info("successfully load model from {}".format(path))

        # Load the processor
        self.processor = AutoProcessor.from_pretrained(path)

    def _inference(self, prompt: PromptStruct, **kwargs) -> str:

        audio = prompt["audio"]
        kwargs["generate_kwargs"] = prompt["generate_kwargs"]

        logger.debug(f"the input is {audio}, {kwargs}")
        pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            torch_dtype=self.torch_dtype,
            device=self.device,
        )
        result = pipe(audio, return_timestamps=True, **kwargs)
        return result["text"]
