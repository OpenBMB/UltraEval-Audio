from typing import Dict, List
from audio_evals.base import PromptStruct
from audio_evals.models.model import Model
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel


class CPM(Model):
    def __init__(self, path: str, sample_params: Dict[str, any] = None):
        super().__init__(True, sample_params)
        model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

        self.model = model.eval().cuda()
        self.model.generation_config.do_sample = False
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)

    def _inference(self, prompt: PromptStruct, **kwargs) -> str:
        text = ""
        for content in prompt:
            if content["role"] == "user":
                for line in content["contents"]:
                    if line["type"] == "text":
                        text = line["value"]

        res = self.model.chat(self.tokenizer, text, **kwargs)
        return res[0]


def _get_stopping_criteria(stop_words, tokenizer, batch_size):
    from transformers import StoppingCriteria, StoppingCriteriaList

    class MultiTokenEOSCriteria(StoppingCriteria):
        """Criteria to stop on the specified multi-token sequence."""

        def __init__(self, stop_words: List[str], tokenizer, batch_size: int):
            self.done_tracker = [False] * batch_size
            self.stop_words, self.max_sequence_id_len = [], 0
            for s in stop_words:
                self.stop_words.append(s)
                sequence_ids = tokenizer.encode(s, add_special_tokens=False)
                self.max_sequence_id_len = max(
                    self.max_sequence_id_len, len(sequence_ids)
                )
            self.tokenizer = tokenizer

        def __call__(self, input_ids, scores, **kwargs) -> bool:
            # compare the last len(stop) tokens
            lookback_ids_batch = input_ids[:, -self.max_sequence_id_len :]
            lookback_tokens_batch = self.tokenizer.batch_decode(lookback_ids_batch)
            for i, done in enumerate(self.done_tracker):
                if done:
                    continue
                self.done_tracker[i] = any(
                    s in lookback_tokens_batch[i] for s in self.stop_words
                )
            return False not in self.done_tracker

    c = MultiTokenEOSCriteria(stop_words, tokenizer, batch_size)
    return StoppingCriteriaList([c])


class CpmPretrain(Model):
    def __init__(self, path: str, sample_params: Dict[str, any] = None):
        super().__init__(False, sample_params)
        model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

        self.model = model.eval().cuda()
        self.model.generation_config.do_sample = False
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)

    def _inference(self, prompt: PromptStruct, **kwargs) -> str:

        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        if "stopping_criteria" in kwargs:
            stopping_criteria = _get_stopping_criteria(
                kwargs["stopping_criteria"], self.tokenizer, inputs.input_ids.shape[0]
            )
            kwargs["stopping_criteria"] = stopping_criteria

        generate_ids = self.model.generate(inputs.input_ids, **kwargs)
        res = self.tokenizer.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return res
