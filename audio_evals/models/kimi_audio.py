from audio_evals.models.model import Model
from audio_evals.base import PromptStruct
from typing import Dict
import torch
import sys
import os
import logging
logger = logging.getLogger(__name__)


import soundfile as sf
cur_path = os.path.abspath(os.path.dirname(__file__))
project_root = os.path.dirname(os.path.dirname(cur_path))
sys.path.append(os.path.join(project_root,"audio_evals/models/"))

from kimia_infer.api.kimia import KimiAudio


class KimiAudioModel(Model):
    def __init__(self, path: str, sample_params: Dict[str, any] = None):
        super().__init__(True, sample_params) 
        logger.debug("start load model from {}".format(path))
        # init model
        self.model = KimiAudio(model_path=path, load_detokenizer=True)
        logger.debug("successfully load model from {}".format(path))    
    
    def _inference(self, prompt: PromptStruct, **kwargs) -> str:
        messages_asr = [
            # You can provide context or instructions as text
            {"role": "user", "message_type": "text", "content": prompt[0]["contents"][0]['value']},
            # Provide the audio file path
            {"role": "user", "message_type": "audio", "content": prompt[0]["contents"][1]['value']}
        ]
        # Generate only text output
        _, response = self.model.generate(messages_asr, **kwargs, output_type="text")
        logger.debug("prompt: {} \nresponse: {}".format(prompt,response))
        return response