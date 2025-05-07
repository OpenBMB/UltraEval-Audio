import platform
from dataclasses import dataclass
import logging
from typing import Union, List, Optional, Tuple
import gc

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as P
from torch.nn.utils.parametrizations import weight_norm
from tqdm import tqdm
from transformers import LlamaModel, LlamaConfig, LogitsWarper
from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.utils import is_flash_attn_2_available
from dataclasses import is_dataclass

# from transformers.generation import TopKLogitsWarper, TopPLogitsWarper


class CustomRepetitionPenaltyLogitsProcessorRepeat:

    def __init__(self, penalty: float, max_input_ids: int, past_window: int):
        if not isinstance(penalty, float) or not (penalty > 0):
            raise ValueError(
                f"`penalty` has to be a strictly positive float, but is {penalty}"
            )

        self.penalty = penalty
        self.max_input_ids = max_input_ids
        self.past_window = past_window

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        if input_ids.size(1) > self.past_window:
            input_ids = input_ids.narrow(1, -self.past_window, self.past_window)
        freq = F.one_hot(input_ids, scores.size(1)).sum(1)
        if freq.size(0) > self.max_input_ids:
            freq.narrow(
                0, self.max_input_ids, freq.size(0) - self.max_input_ids
            ).zero_()
        alpha = torch.pow(self.penalty, freq)
        scores = scores.contiguous()
        inp = scores.multiply(alpha)
        oth = scores.divide(alpha)
        con = scores < 0
        out = torch.where(con, inp, oth)
        del inp, oth, scores, con, alpha
        return out


def del_all(d: Union[dict, list]):
    if is_dataclass(d):
        for k in list(vars(d).keys()):
            x = getattr(d, k)
            if isinstance(x, dict) or isinstance(x, list) or is_dataclass(x):
                del_all(x)
            del x
            delattr(d, k)
    elif isinstance(d, dict):
        lst = list(d.keys())
        for k in lst:
            x = d.pop(k)
            if isinstance(x, dict) or isinstance(x, list) or is_dataclass(x):
                del_all(x)
            del x
    elif isinstance(d, list):
        while len(d):
            x = d.pop()
            if isinstance(x, dict) or isinstance(x, list) or is_dataclass(x):
                del_all(x)
            del x
    else:
        del d


class GPT(nn.Module):
    def __init__(
        self,
        gpt_config: dict,
        num_audio_tokens: int = 626,
        num_text_tokens: int = 21178,
        num_vq=4,
        attn_implementation="sdpa",
    ):
        super().__init__()

        self.num_vq = num_vq
        self.num_audio_tokens = num_audio_tokens
        self.attn_implementation = attn_implementation

        llama_config = LlamaConfig(
            **gpt_config, attn_implementation=attn_implementation
        )

        model = LlamaModel(llama_config)
        # del model.embed_tokens # 不删除 model.embed_tokens 因为他能指示模型的device
        self.gpt = model

        self.model_dim = int(self.gpt.config.hidden_size)

        # audio codes的编码头，一共4个
        self.emb_code = nn.ModuleList(
            [
                nn.Embedding(
                    num_audio_tokens,
                    self.model_dim,
                )
                for _ in range(num_vq)
            ],
        )
        self.emb_text = nn.Embedding(num_text_tokens, self.model_dim)

        # audio codes的解码头，一共4个
        self.head_code = nn.ModuleList(
            [
                weight_norm(
                    nn.Linear(
                        self.model_dim,
                        num_audio_tokens,
                        bias=False,
                    ),
                    name="weight",
                )
                for _ in range(self.num_vq)
            ],
        )

    def from_pretrained(self, file_path: str):
        self.load_state_dict(torch.load(file_path, weights_only=True, mmap=True))
