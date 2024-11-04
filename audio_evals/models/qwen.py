import logging
import re
from copy import deepcopy
from typing import Dict

import librosa
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

from audio_evals.base import PromptStruct
from audio_evals.models.model import Model

logger = logging.getLogger(__name__)


def process_prompts(prompt: PromptStruct):
    def _conv_contents(content):
        content = deepcopy(content)
        for ele in content:
            if ele["type"] == "audio":
                ele["audio_url"] = ele["value"]
            elif ele["type"] == "text":
                ele["text"] = ele["value"]
            del ele["value"]
        return content

    for line in prompt:
        line["content"] = _conv_contents(line["contents"])
        del line["contents"]

    return prompt


class Qwen2audio(Model):
    def __init__(self, path: str, sample_params: Dict[str, any] = None):
        super().__init__(True, sample_params)
        logger.debug("start load model from {}".format(path))
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
            path,
            device_map="cuda",
            trust_remote_code=True,
        ).eval()
        logger.debug("successfully load model from {}".format(path))
        self.processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)

    def _inference(self, prompt: PromptStruct, **kwargs):
        prompt = process_prompts(prompt)
        text = self.processor.apply_chat_template(
            prompt, add_generation_prompt=True, tokenize=False
        )
        audios = []
        for message in prompt:
            if isinstance(message["content"], list):
                for ele in message["content"]:
                    if ele["type"] == "audio":
                        audios.append(
                            librosa.load(
                                ele["audio_url"],
                                sr=self.processor.feature_extractor.sampling_rate,
                            )[0]
                        )
        prompt = [
            {"role": "system", "content": "You are a helpful assistant."}
        ] + prompt
        logger.debug("prompt: {}".format(prompt))
        inputs = self.processor(
            text=text, audios=audios, return_tensors="pt", padding=True
        )
        for k, v in inputs.items():
            inputs[k] = v.to("cuda")
        inputs.input_ids = inputs.input_ids.to("cuda")
        generate_ids = self.model.generate(**inputs, **kwargs)
        generate_ids = generate_ids[:, inputs.input_ids.size(1) :]

        response = self.processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return response


class Qwen2audioPretrain(Model):
    def __init__(self, path: str, sample_params: Dict[str, any] = None):
        super().__init__(False, sample_params)
        logger.debug("start load model from {}".format(path))
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
            path,
            device_map="cuda",
            trust_remote_code=True,
        ).eval()
        logger.debug("successfully load model from {}".format(path))
        self.processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)

    def _inference(self, prompt: PromptStruct, **kwargs):
        match = re.search(
            r"<\|audio_bos\|><\|(.*?)\|><\|audio_eos\|>", prompt, re.DOTALL
        )
        assert match, "no audio file found in prompt"
        f_name = match.group(1)
        prompt = prompt.replace(f_name, "AUDIO")
        audio, sr = librosa.load(
            f_name, sr=self.processor.feature_extractor.sampling_rate
        )
        inputs = self.processor(text=prompt, audios=audio, return_tensors="pt")

        for k, v in inputs.items():
            inputs[k] = v.to("cuda")
        inputs.input_ids = inputs.input_ids.to("cuda")
        generated_ids = self.model.generate(**inputs, **kwargs)
        generated_ids = generated_ids[:, inputs.input_ids.size(1) :]
        response = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return response
