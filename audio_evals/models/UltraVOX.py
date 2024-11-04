# pip install transformers peft librosa
import logging
from typing import List, Dict, Tuple

import transformers
import librosa

from audio_evals.base import PromptStruct
from audio_evals.models.model import Model

logger = logging.getLogger(__name__)


class UltraVOX(Model):
    def __init__(self, path: str, sample_params: Dict[str, any] = None):
        super().__init__(True, sample_params)  # as a chat model
        logger.debug("start load model from {}".format(path))
        self.pipe = transformers.pipeline(model=path, trust_remote_code=True, device=0)
        logger.debug("model loaded")
        self.max_new_tokens = 30

    @staticmethod
    def _conv_prompt(prompt: PromptStruct) -> Tuple[str, str, List[Dict[str, str]]]:
        audio, sr = "", ""
        turns = [
            {
                "role": "system",
                "content": "You are a friendly and helpful character. You love to answer questions for people.",
            },
        ]
        for line in prompt:
            role = line["role"]
            for c in line["contents"]:
                if c["type"] == "audio":
                    audio, sr = librosa.load(c["value"], sr=16000)
                if c["type"] == "text":
                    turns.append({"role": role, "content": c["value"] + " <|audio|>"})
        return audio, sr, turns

    def _inference(self, prompt: PromptStruct, **kwargs) -> str:
        audio, sr, turns = self._conv_prompt(prompt)
        logger.debug("turns: {}".format(turns))
        return self.pipe(
            {"audio": audio, "turns": turns, "sampling_rate": sr}, **kwargs
        )
