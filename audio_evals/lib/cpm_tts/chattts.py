import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
import random

from .config import Config
from .gpt import GPT
from .dvae import DVAE
from .minicpmv26_resampler import Resampler
from vocos import Vocos
from vocos.pretrained import instantiate_class

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as P
from torch.nn.utils.rnn import pad_sequence

from transformers import BertTokenizerFast
from transformers import PreTrainedModel, PretrainedConfig

import torch
import random
import numpy as np
import math


def apply_spk_emb(
    input_ids: torch.Tensor = None,  # [batch_size, seq_len_max]
    spk_emb: torch.Tensor = None,  # [batch_size, num_spk_emb, hidden_dim]
    input_embeds: torch.Tensor = None,  # [batch_size, seq_len_max, hidden_dim]
    spk_emb_token_id: int = 0,
    num_spk_embs: int = 1,
):
    """
    把Inputs_embeds里面的连续 spk_emb placeholder 替换为 预先准备好的 spk_emb (注意：是in place替换，不是创建新tensor，所以不会返回tensor)
    """
    # 定义 input_embeds, input_ids, spk_emb
    # batch_size = 3
    # seq_len = 5
    # hidden = 4

    # input_embeds = torch.randn(batch_size, seq_len, hidden)
    # input_ids = torch.tensor([
    #     [1, 100, 3, 4, 100],
    #     [100, 2, 3, 100, 5],
    #     [1, 2, 100, 4, 5]
    # ])
    # spk_emb = torch.randn(batch_size, hidden)

    # 对spk_emb进行正则化 省去了 因为前序已经对spk_emb做了 l2 norm
    # spk_emb = F.normalize(spk_emb, p=2, dim=1)
    batch_size = input_ids.shape[0]

    for idx in range(batch_size):
        # 1. 生成掩码，标记 input_ids 中等于100的位置
        input_ids_ = input_ids[idx]  # [seq_len_max]
        spk_emb_ = spk_emb[idx]  # [num_spk_emb]
        # input_embed_ = input_embeds[idx] # [seq_len_max, hidden_dum]
        mask_ = input_ids_ == spk_emb_token_id  # [batch_size, seq_len_max]
        nonzero_position_idx = mask_.nonzero(as_tuple=False)  # [num_spk_emb, 1]
        assert nonzero_position_idx.shape[0] == num_spk_embs
        begin_idx = nonzero_position_idx.min()
        end_idx = nonzero_position_idx.max()
        input_embeds[idx, begin_idx : end_idx + 1, :] = spk_emb_

        # print(f"begin: {begin_idx}, end: {end_idx}")

    # 2. 扩展掩码维度以匹配 input_embeds
    # mask = mask.unsqueeze(-1)  # [batch_size, seq_len, 1]

    # 3. 扩展 spk_emb 以便进行广播
    # spk_emb_expanded = spk_emb.unsqueeze(1)  # [batch_size, 1, hidden]

    # 4. 使用掩码将 input_embeds 中对应位置替换为 spk_emb
    # modified_input_embeds = torch.where(mask, spk_emb_expanded, input_embeds)

    return


def make_streaming_chunk_mask(
    input_embeds: torch.Tensor,  # [batch_size, seq_len, hidden_dim] 合并文本、音频后的input_embeds
    tts_text_scopes: List[List[int]],  # List[List[int, int]]
    tts_audio_scopes: List[List[int]],  # List[List[int, int]]
    tts_text_masks: List[torch.Tensor],  # List[Tensor[seq_len_max]]
    min_chunk_num_token: int = 5,  # 每 chunk 音频，模型能看到的最少的 新的文本token数量
    max_chunk_num_token: int = 7,  # 每 chunk 音频，模型能看到的最多的 新的文本token数量
    tts_chunk_len_s: int = 50,  # 音频 chunk 的大小，50 大约对应 1s 时间的音频
):
    """
    创建一个 look-ahead chunked attention mask ，能够让 tts transformer 在每生成第 N~N+1 秒音频时只看到前 M 个token 进而实现流式 TTS.

    Input sequence: [t1, t2, t3, t4, t5, [Ptts], a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, ...]
    Output: 4d causal mask

    ------- text positions -------
    [0]
    [0, 0]
    [0, 0, 0]
    [0, 0, 0,    0]
    [0, 0, 0,    0,    0]
    [0, 0, 0,    0,    0,    0] <- here is [Ptts]
    ------- audio positions --------
                             v- here is [Ptts]
    [0, 0, -inf, -inf, -inf, 0, 0]
    [0, 0, -inf, -inf, -inf, 0, 0, 0]
    [0, 0, -inf, -inf, -inf, 0, 0, 0, 0]
    [0, 0, -inf, -inf, -inf, 0, 0, 0, 0, 0]
    [0, 0, -inf, -inf, -inf, 0, 0, 0, 0, 0, 0] # end of first 1s audio chunk
    [0, 0, 0   , -inf, -inf, 0, 0, 0, 0, 0, 0, 0]
    [0, 0, 0   , -inf, -inf, 0, 0, 0, 0, 0, 0, 0, 0]
    [0, 0, 0   , -inf, -inf, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    [0, 0, 0   , -inf, -inf, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    [0, 0, 0   , -inf, -inf, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    """

    # 构建一个完整的 input embeds 的 attention mask [batch_size, seq_len], 当然这里没有考虑audio的mask，因为audio总是在后面。
    batch_size = input_embeds.shape[0]
    input_embeds_attention_mask = torch.ones(
        input_embeds.shape[0],
        input_embeds.shape[1],
        dtype=torch.int8,
        device=input_embeds.device,
    )

    for idx in range(batch_size):
        input_embeds_attention_mask[
            idx, tts_text_scopes[idx][0] : tts_text_scopes[idx][1]
        ] = tts_text_masks[idx]

    # 初始化一个标准的上三角causal mask
    dtype = input_embeds.dtype
    device = input_embeds.device
    # dtype = torch.bfloat16
    min_dtype = torch.finfo(dtype).min
    sequence_length = input_embeds.shape[1]  # [batch_size, seq_len, dim]
    causal_mask = torch.full(
        (sequence_length, sequence_length),
        fill_value=min_dtype,
        dtype=dtype,
        device=device,
    )
    if sequence_length != 1:
        causal_mask = torch.triu(causal_mask, diagonal=1)
    else:
        raise ValueError("sequence_length of tts could not be 1.")
    causal_mask = causal_mask.unsqueeze(0).repeat(input_embeds.shape[0], 1, 1)

    # 对于每个数据
    for idx in range(input_embeds.shape[0]):
        tts_audio_scope = tts_audio_scopes[idx]
        tts_text_scope = tts_text_scopes[idx]

        audio_token_start = tts_audio_scope[0]
        audio_duration = tts_audio_scope[1] - tts_audio_scope[0]

        # 记录当前 audio chunk 能看到哪个 text chunk 之前的所有 text
        # print("audio_token_start", audio_token_start)
        # print("audio_duration", audio_duration)
        text_pivot = 0

        # 对于每个 chunk 的 audio
        # print("audio number of chunks", math.ceil(audio_duration / tts_chunk_len_s))
        for chunk_idx in range(math.ceil(audio_duration / tts_chunk_len_s)):
            # print("     audio chunk id", chunk_idx)
            audio_chunk_start = audio_token_start + chunk_idx * tts_chunk_len_s
            audio_chunk_end = audio_token_start + (chunk_idx + 1) * tts_chunk_len_s
            # print("     audio_chunk_start", audio_chunk_start)
            # print("     audio_chunk_end", audio_chunk_end)
            # 这一个 新 audio chunk 新看到的 text chunk
            new_text_this_chunk = random.randint(
                min_chunk_num_token, max_chunk_num_token
            )
            text_pivot += new_text_this_chunk
            # print("     new_text_this_chunk", new_text_this_chunk)
            # print("     text_pivot", text_pivot)
            # print("tts_text_scope[0] + text_pivot", tts_text_scope[0] + text_pivot, "tts_text_scope[1]", tts_text_scope[1])
            # 把能看到的text chunk之后所有text chunk mask掉，但不包括[Ptts] token 所以 tts_text_scope[1]+1-1 把不该看到的部分mask掉
            # print("audio_chunk_start, audio_chunk_end, tts_text_scope[0] + text_pivot, tts_text_scope[1]", audio_chunk_start, audio_chunk_end, tts_text_scope[0] + text_pivot, tts_text_scope[1])
            causal_mask[
                idx,
                audio_chunk_start:audio_chunk_end,
                tts_text_scope[0] + text_pivot : tts_text_scope[1],
            ] = min_dtype  # 这是一块方形的区域

        # 还应该把tts_text_masks中为0的部分（padding部分）也mask掉（没有任何一个位置会注意到它）并且，因为text部分在tts里不算loss，所以，也不用考虑label了。
        tts_text_mask_ = input_embeds_attention_mask[idx]
        # print("tts_text_mask.shape", tts_text_mask_.shape)
        # print("tts_text_mask", tts_text_mask_)
        causal_mask[idx, :, tts_text_mask_ == 0] = min_dtype
        # print("tts_text_mask", tts_text_mask_ == 0)

    causal_mask = causal_mask.unsqueeze(1)  # [batch_size, 1, seq_len, seq_len]

    return causal_mask


class LLMConditionalChatTTS(PreTrainedModel):
    """
    能够接受 LLM 输出的 hidden state 和文本 input_ids 的 TTS 模型，输出 audio codes
    """

    def __init__(
        self,
        config: Config,
        attn_implementation: str = "sdpa",
    ):
        super().__init__(PretrainedConfig())

        # 把 llm 的hidden state 投射到 self.decoder.hideen_state
        # 选择使用 Linear 还是 Resampler 作为 projector
        if config.llm_projector.use_resampler:
            llm_projector = Resampler(
                num_queries=config.llm_projector.query_num,
                embed_dim=config.gpt.hidden_size,
                num_heads=config.gpt.hidden_size // 128,
                kv_dim=config.llm_projector.llm_dim,
                max_size=(1, config.llm_projector.max_size),
            )
        else:
            llm_projector = nn.Linear(
                config.llm_projector.llm_dim, config.gpt.hidden_size, bias=False
            )

        # 把llm的hidden state project到chattts.gpt(只针对spk_emb)
        spk_emb_projector = nn.Linear(
            config.llm_projector.llm_dim, config.gpt.hidden_size, bias=False
        )

        dvae = DVAE(
            decoder_config=asdict(config.dvae.decoder),
            encoder_config=asdict(config.dvae.encoder),
            vq_config=asdict(config.dvae.vq),
            dim=config.dvae.decoder.idim,
            coef=None,
        )

        # 这个GPT模型是个Llama model 输入文本input_ids可以解码音频codes
        decoder = GPT(
            gpt_config=asdict(config.gpt),
            attn_implementation=attn_implementation,
        )

        # self.vocos = vocos
        self.spk_emb_projector = spk_emb_projector
        self.llm_projector = llm_projector
        self.dvae = dvae
        self.decoder = decoder

        self.pooling = config.pooling
        self.use_speaker_embedding = config.use_speaker_embedding
        self.use_llm_hidden_state = config.use_llm_hidden_state
        self.num_spk_embs = config.num_spk_embs
        self.spk_emb_token_id = config.spk_emb_token_id

        # self.stts_token_id = stts_token_id
        # self.ptts_token_id = ptts_token_id
        # self.spk_emb_token_id = spk_emb_token_id
        # self.num_spk_embs = num_spk_embs

        self.streaming = config.streaming
        self.streaming_text_chunk_min = config.streaming_text_chunk_min
        self.streaming_text_chunk_max = config.streaming_text_chunk_max
        self.tts_chunk_len_s = config.tts_chunk_len_s

        self.use_text = config.use_text

        self.config = config

        return

    def load(self, path, **kwargs):
        dvae_ckpt_path: str = os.path.join(path, "DVAE_full.pt")
        gpt_ckpt_path: str = os.path.join(path, "GPT.pt")

        # vocos_ckpt_path: str = os.path.join(path, "Vocos.pt")
        # print("loading vocos checkpoint")
        # self.vocos.load_state_dict(torch.load(vocos_ckpt_path, weights_only=True, mmap=True), **kwargs)
        # print("vocos checkpoint load ok!")

        strict = kwargs.get("strict", False)

        print("loading dvae checkpoint")
        self.dvae.load_state_dict(torch.load(dvae_ckpt_path), strict=strict)
        print("dvae checkpoint load ok!")

        random_decoder = kwargs.get("random_decoder", False)
        if random_decoder:
            print("randomly initilaize tts decoder.")
            self.decoder.gpt._init_weights(self.decoder)
        else:
            print("load decoder (llama) checkpoints")
            state_dict_result = self.decoder.load_state_dict(
                torch.load(gpt_ckpt_path), strict=strict
            )
            if state_dict_result.missing_keys:
                print("TTS missing keys:")
                for key in state_dict_result.missing_keys:
                    print(key)

            if state_dict_result.unexpected_keys:
                print("TTS unexpected keys:")
                for key in state_dict_result.unexpected_keys:
                    print(key)
            print("decoder (llama) checkpoint load ok!")

        return

    def forward(
        self,
        input_ids,  # List[Tensor[seq_len]]
        lm_spk_emb_last_hidden_states=None,  # List[Tensor[gpt_dim]], here each lm_spk_emb_last_hidden_states `requires` gradient
        lm_last_hidden_states=None,  # List[Tensor[seq_len_0, gpt_dim]], here each lm_last_hidden_state `requires` gradient
        target_audio_features=None,  # List[Tensor[num_channels, num_samples]]
        streaming_tts_text_masks=None,  # List[Tensor[seq_len_max]]
        **kwargs,
    ):
        """
        batch 中每个模型应该说话的区域的 LLM last hidden state (从LLM取出来，带梯度) 和 说的文本 Ground truth (没有梯度) 和 目标音频 (没有梯度)
        计算TTS损失，并把梯度反传到lm_last_hidden_states的每个元素中。

        - 10/3: 支持空的输入，就是dummy train，在没有音频的任务中有用，可以防止训练因为部分参数没有被用到而卡住。
        - 10/11: 支持 eos token

        Args:
            input_ids: 每个模型应该说话的区域的 文本ground truth 的 input_ids，其中所有的元素都是Tensor，每个Tensor都不定长
            lm_spk_emb_last_hidden_states: 来自语言模型的 spk_emb last hidden state。
            lm_last_hidden_states: 每个模型应该说话的区域的 LLM last hidden state组成的List，其中所有的元素都是Tensor，每个Tensor都不定长。
            target_audio_features: 每个模型应该说话的区域的 mel ground truth, 其中所有的元素都是Tensor，每个Tensor都不定长。
            streaming_tts_text_masks: 在 streaming 训练下，需要把 tts text pad 成定长，所以产生了 text mask。形状为 Tensor[seq_len_max]
        """

        dummy = False
        # 考虑 dummy train 的情况
        if self.train:
            if len(input_ids) == 0:
                dummy = True
                dummy_seq_len = 100
                input_ids = [
                    torch.full(
                        (dummy_seq_len,),
                        fill_value=1,
                        device=self.decoder.gpt.embed_tokens.weight.device,
                        dtype=torch.int64,
                    )
                ]
                input_ids[0][
                    0 : self.num_spk_embs
                ] = self.spk_emb_token_id  # 我们需要每个参数都用到...

                lm_spk_emb_last_hidden_states = [
                    torch.full(
                        (self.num_spk_embs, self.config.llm_projector.llm_dim),
                        fill_value=0,
                        device=self.decoder.gpt.embed_tokens.weight.device,
                        dtype=self.decoder.gpt.embed_tokens.weight.dtype,
                    )
                ]  # 我们需要每个参数都用到...

                lm_last_hidden_states = [
                    torch.full(
                        (100, self.config.llm_projector.llm_dim),
                        fill_value=0,
                        device=self.decoder.gpt.embed_tokens.weight.device,
                        dtype=self.decoder.gpt.embed_tokens.weight.dtype,
                    )
                ]

                target_audio_features = [
                    torch.full(
                        (100, 100),
                        fill_value=0,
                        device=self.decoder.gpt.embed_tokens.weight.device,
                        dtype=self.decoder.gpt.embed_tokens.weight.dtype,
                    )
                ]
                streaming_tts_text_masks = None

        generate = kwargs.get("generate", False)

        # 如果传入了 lm_last_hidden_states 就需要构造 condition
        if lm_last_hidden_states is not None:
            # 1. project llm last hidden states (QwenAudio, Qwen2) to tts gpt decoder hidden size (as tts condition) first
            # 1.1. keep trace of length of each tts condition
            assert len(lm_last_hidden_states) != 0
            all_tts_condition_seq_len = [i.shape[0] for i in lm_last_hidden_states]

            # 1.2.1 pad hidden states to be a big tensor for high efficiency
            # all_lm_last_hidden_states = pad_sequence(lm_last_hidden_states, batch_first=True) # [batch_size, seq_len_max, lm_hidden_size]
            # 1.2.1 计算每个样本的 (height, width)，对于 Resampler 来说，height 固定为 1，width 是序列长度
            tgt_sizes = torch.tensor(
                [
                    (1, lm_hidden_state.size(0))
                    for lm_hidden_state in lm_last_hidden_states
                ]
            )

            # 1.2.2 pad hidden states to be a big tensor for high efficiency ---- [batch_size, seq_len_max, lm_hidden_size]
            input_data = pad_sequence(lm_last_hidden_states, batch_first=True)

            # 1.2.3 all_lm_last_hidden_states ->  all_tts_conditions
            # if isinstance(self.llm_projector, Resampler):
            if self.config.llm_projector.use_resampler:
                all_tts_condition = self.llm_projector(
                    input_data, tgt_sizes
                )  # [batch_size, llm_projector.query_num=1, gpt_hidden_size]
            else:
                all_tts_condition = self.llm_projector(
                    input_data
                )  # [batch_size, seq_len_max, gpt_hidden_size]

            # 进行normalize
            all_tts_condition = F.normalize(
                all_tts_condition, p=2, dim=2
            )  # 进行L2 norm # [batch_size, seq_len_max, gpt_hidden_size]

            # 1.3. split whole tensor into list[Tensor] and remove padding position
            all_tts_condition_varlen = []
            for idx in range(all_tts_condition.shape[0]):
                all_tts_condition_varlen.append(
                    all_tts_condition[idx, 0 : all_tts_condition_seq_len[idx]]
                )

            # 如果需要把 condition 进行 pooling
            if self.pooling:
                raise NotImplementedError
                pooled = []
                for c in all_tts_condition_varlen:
                    pooled.append(torch.mean(c, dim=0))  # 维度: 768 -> 768
                # 这时需要把 condition 的 seq_len 都改成1
                all_tts_condition_seq_len = [1 for _ in all_tts_condition_varlen]
                # 进行 normalize
                pooled = torch.stack(pooled, dim=0)  # [batch, gpt_dim]
                pooled_normalized = F.normalize(pooled, p=2, dim=1)  # 进行L2 norm
                # 覆盖 all_tts_condition_varlen
                # [batch, gpt_dim] 拆分成 List[tensor[gpt_dim]]
                all_tts_condition_varlen = []
                for i in range(pooled_normalized.shape[0]):
                    all_tts_condition_varlen.append(
                        pooled_normalized[i].unsqueeze(0)
                    )  # [gpt_dim] -> [seq_len=1, gpt_dim]
        else:
            all_tts_condition_varlen = None

        # 如果传入了 spk_emb 的last hidden state，需要把 spk_emb 替换进 input_embeds
        if (
            lm_spk_emb_last_hidden_states is not None
        ):  # List[Tensor[num_spk_emb, lm_hidden_dim]]
            if len(lm_spk_emb_last_hidden_states) == 0:
                raise ValueError("lm_spk_emb_last_hidden_states is empty.")
            # [bs, num_spk_emb, lm_hidden_dim] 如果每个数据 spk_emb 不等量这里会报错
            stacked_lm_spk_emb_last_hidden_states = torch.stack(
                lm_spk_emb_last_hidden_states, dim=0
            )
            # 检查 num_spk_embs 数量是否和预期一样
            assert stacked_lm_spk_emb_last_hidden_states.shape[1] == self.num_spk_embs
            # 统一project到tts decoder维度
            gpt_spk_emb_last_hidden_states = self.spk_emb_projector(
                stacked_lm_spk_emb_last_hidden_states
            )  # [bs, num_spk_emb, gpt_dim]
            # 标准化
            gpt_spk_emb_last_hidden_states = F.normalize(
                gpt_spk_emb_last_hidden_states, p=2, dim=-1
            )  # 进行L2 norm [batch_size, num_spk_emb hidden_dim]
            # 拆分 batch tensor 为List[Tensor]
            gpt_spk_emb_last_hidden_states_varlen = [
                i for i in gpt_spk_emb_last_hidden_states
            ]  # List[Tensor[num_spk_emb, lm_hidden_dim]]
        else:
            gpt_spk_emb_last_hidden_states_varlen = None

        # 这意味着正在训练，现场音频编码 encode audio waveforms to audio tokens using dVAE
        if target_audio_features is not None:
            eos_token_id = int(self.decoder.emb_code[0].num_embeddings - 1)
            self.dvae.eval()
            with torch.no_grad():  # 这里很遗憾没有做成 batch 处理，因为我们不知道 chattts 会不会在训练时 pad mel 谱
                all_audio_codes = []
                # 语音的话也许保留 float32 编码是必要的，即使慢了一些
                with torch.cuda.amp.autocast(dtype=torch.float):
                    for audio_waveform in target_audio_features:
                        audio_codes = self.dvae(
                            audio_waveform, mode="encode"
                        )  # Tensor[1, 4, audio_seq_len]
                        # 加入 eos token
                        audio_codes_with_eos = torch.cat(
                            (
                                audio_codes.squeeze(0),  # [num_vq, seq_len]
                                torch.ones(
                                    self.decoder.num_vq,
                                    1,
                                    device=audio_codes.device,
                                    dtype=audio_codes.dtype,
                                )
                                * eos_token_id,  # [num_vq, 1]
                            ),
                            dim=-1,
                        )
                        all_audio_codes.append(
                            audio_codes_with_eos
                        )  # Tensor[4, audio_seq_len]

            all_audio_codes_seq_len = [i.shape[1] for i in all_audio_codes]

            # 按层，编码 4 层 codes 到 audio embedding
            audio_embed_all_layers = []
            for i in range(self.decoder.num_vq):
                audio_codes_layer_i = []
                for codes in all_audio_codes:
                    # print("codes.shape", codes.shape)
                    # print("codes[i, :].squeeze(0).shape", codes[i, :].squeeze(0).shape)
                    audio_codes_layer_i.append(
                        codes[i, :].squeeze(0),
                    )
                # pad 每一层audio codes 成定长
                audio_codes_layer_i = pad_sequence(
                    audio_codes_layer_i, batch_first=True
                )
                # 把每一层 audio codes 编码成 embedding (并行化)
                audio_embed_layer_i = self.decoder.emb_code[i](
                    audio_codes_layer_i
                )  # [batch_size, seq_len, gpt_hidden_dim]
                audio_embed_all_layers.append(audio_embed_layer_i)

            # 这里需要计算四层的 audio_embed 然后加起来
            # 按照ChatTTS官方实现 https://github.com/2noise/ChatTTS/blob/51ec0c784c2795b257d7a6b64274e7a36186b731/ChatTTS/model/gpt.py#L451
            audio_embed_all_layers = torch.stack(
                audio_embed_all_layers, dim=0
            )  # [num_vq, seq_len, gpt_hidden_dim]
            audio_embed_all_layers = torch.sum(
                audio_embed_all_layers, dim=0, keepdim=False
            )  # [seq_len, gpt_hidden_dim]

            # 根据存储的 audio codes 的原始长度变回不定长序列
            audio_embed_all_layers_varlen = []
            for idx in range(audio_embed_all_layers.shape[0]):
                audio_embed_all_layers_varlen.append(
                    audio_embed_all_layers[idx, 0 : all_audio_codes_seq_len[idx]]
                )

        # 编码 TTS 文本部分变成 embeds
        all_input_ids_seq_len = [i.shape[0] for i in input_ids]
        input_ids = pad_sequence(input_ids, batch_first=True)
        all_text_embeds = self.decoder.emb_text(
            input_ids
        )  # [batch_size, seq_len] -> [batch_size, seq_len, gpt_hidden_dim]

        # 融合 spk_emb: 如果传入了 spk_emb，需要替换到 embeds 里
        if lm_spk_emb_last_hidden_states is not None:
            if generate:
                all_text_embeds_old = all_text_embeds.clone()
            # 这里是 in place 替换 all_text_embeds 里的一些位置为 spk emb
            apply_spk_emb(
                input_ids=input_ids,
                spk_emb=gpt_spk_emb_last_hidden_states,
                input_embeds=all_text_embeds,
                spk_emb_token_id=self.spk_emb_token_id,
                num_spk_embs=self.num_spk_embs,
            )
            # 检查一下是否成功替换
            if generate:
                with torch.no_grad():
                    diff = torch.mean(
                        torch.abs(all_text_embeds_old - all_text_embeds)
                    ).item()
                    print("diff between all_text_embeds_old & all_text_embeds", diff)
                    assert diff > 0

        all_text_embeds_varlen = []
        # 变回变长序列 方便后续不同 token 融合
        for idx in range(all_text_embeds.shape[0]):
            all_text_embeds_varlen.append(
                all_text_embeds[idx, 0 : all_input_ids_seq_len[idx], :]
            )  # List[ Tensor[seq_len, gpt_hidden_dim] ]
        # 融合 tts condition 和 audio codes 和text token的embeds，应该不需要大量计算 不做成batch应该可以接受，不会中断梯度

        # 最终拼接的格式
        # tts condition from llm last hidden state(can be pooling) | embeds of [Stts] [spk_emb] text_embeds embeds of [Ptts] | audio embeds

        # 合并多个来源的 embeds
        embeds_to_merge = []

        # 加入 lm condition
        if lm_last_hidden_states is not None:
            embeds_to_merge.append(all_tts_condition_varlen)

        # 加入 text
        if self.use_text:
            embeds_to_merge.append(all_text_embeds_varlen)

        # 如果传入了 audio feature，加入 audio embeds
        if target_audio_features is not None:
            embeds_to_merge.append(audio_embed_all_layers_varlen)

        # 合并 embeds
        all_merged_embeds_ = []
        for item_tuple in zip(*embeds_to_merge):
            merged_embed = torch.cat(
                item_tuple, dim=0
            )  # [seq_len_tts_condition+seq_len_text+seq_len_audio, gpt_hidden_dim]
            all_merged_embeds_.append(merged_embed)

        input_embeds_seqlen = []
        for i in all_merged_embeds_:
            input_embeds_seqlen.append(i.shape[0])

        # 这里会把每个序列 embeds 给 pad，形成一个整齐的 tensor，因为马上要送入 transformer 了
        # 这里也不产生 attention mask 因为我们用了右 padding
        input_embeds = pad_sequence(
            all_merged_embeds_, batch_first=True
        )  # List[ Tensor[seq_len_i, gpt_hidden_dim] ] -> Tensor[batch_size, seq_len_max, gpt_hidden_dim]

        # 确定每个数据中文本所在位置
        text_ranges = []
        batch_size = input_embeds.shape[0]
        for idx in range(batch_size):
            start_idx = 0

            # 如果传入了 hidden state 需要考虑 hidden state 的长度
            if lm_last_hidden_states is not None:
                start_idx += all_tts_condition_seq_len[idx]

            end_idx = start_idx + all_input_ids_seq_len[idx]
            text_ranges.append((start_idx, end_idx))

        if target_audio_features is not None:
            batch_size = input_embeds.shape[0]
            seq_len_max = input_embeds.shape[1]

            # 我们在这里构建一个 labels，只有 audio codes 位置会被学习。[batch_size, seq_len, num_vqs]
            labels = torch.zeros(
                batch_size,
                seq_len_max,
                self.decoder.num_vq,
                device=input_embeds.device,
                dtype=torch.long,
            )
            labels[:, :, :] = -100

            # 确定每个数据中 audio codes 所在位置
            audio_codes_ranges = []
            for idx in range(batch_size):
                start_idx = 0
                # 如果传入了 hidden state 需要考虑 hidden state 的长度
                if lm_last_hidden_states is not None:
                    start_idx += all_tts_condition_seq_len[idx]

                if self.use_text:
                    start_idx += all_input_ids_seq_len[idx]

                end_idx = start_idx + all_audio_codes_seq_len[idx]
                audio_codes_ranges.append((start_idx, end_idx))

            # 把音频 labels 替换进 labels
            for idx, audio_codes_range in zip(range(batch_size), audio_codes_ranges):
                start_idx = audio_codes_range[0]
                end_idx = audio_codes_range[1]
                labels[idx, start_idx:end_idx, :] = all_audio_codes[idx].permute(1, 0)

            # For REAL streaming ChatTTS setting, a simple way is to create a self-defined 4D attention mask to the model, then we can control which kv can be attended by which q.
            # https://github.com/huggingface/transformers/blob/65bb28444849976f853063edb958b3ef3dd59d12/src/transformers/models/llama/modeling_llama.py#L59
            # It says, `Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.`

            if self.streaming and not dummy:
                tts_attention_mask_4d = make_streaming_chunk_mask(
                    input_embeds=input_embeds,  # 合并文本、音频后的 input_embeds
                    tts_text_scopes=text_ranges,  # List[Tuple[int, int]]
                    tts_audio_scopes=audio_codes_ranges,  # List[Tuple[int, int]]
                    tts_text_masks=streaming_tts_text_masks,  # List[Tensor[seq_len_max]]
                    min_chunk_num_token=self.streaming_text_chunk_min,
                    max_chunk_num_token=self.streaming_text_chunk_max,
                    tts_chunk_len_s=self.tts_chunk_len_s,
                )  # [batch_size, 1, seq_len, seq_len]
            else:
                tts_attention_mask_4d = None

            # invoke gpt forward AND get last hidden states AND predict audio codes
            # print("input_embeds.shape", input_embeds.shape)

            outputs = self.decoder.gpt(  # self.decoder.gpt is a Llama model, not LlamaForCausalLM
                inputs_embeds=input_embeds,
                attention_mask=tts_attention_mask_4d,
                # attention_mask=attention_mask, # here we don't use attention mask because we use right padding, and we have manually made labels know where should learn
            )

            tts_last_hidden_state = (
                outputs.last_hidden_state
            )  # [batch, seq_len_max, gpt_hidden_dim]

            # predict audio codes using last_hidden_state by gpt TTS decoder
            # 这里是 batch 的
            logits_all_vq_layers = []
            for num_vq_iter in range(self.decoder.num_vq):
                logits_i = self.decoder.head_code[num_vq_iter](
                    tts_last_hidden_state
                )  # [batch, seq_len_max, audio_codebook_vocab]
                logits_all_vq_layers.append(logits_i)
            logits_all_vq_layers = torch.stack(
                logits_all_vq_layers, dim=0
            )  # [num_vq, batch_size, seq_len_max, audio_codebook_vocab], stack, insert one extra dimension
            logits_all_vq_layers = logits_all_vq_layers.permute(
                1, 2, 0, 3
            )  # [batch_size, seq_len_max, num_vq, audio_codebook_vocab]

            # compute model predictions
            shift_logits = logits_all_vq_layers[
                :, :-1, :, :
            ].contiguous()  # [batch_size, seq_len_max-1, num_vq, audio_codebook_vocab]
            shift_labels = labels[
                :, 1:, :
            ].contiguous()  # [batch_size, seq_len_max-1, num_vq]

            if dummy:
                print("dummy forward!")
                # 考虑到如果这是一个 dummy 训练的话
                # shift_labels[:, :, :] = -100

            # shift_logits = logits_all_vq_layers[:, :-1, 0, :].contiguous() # [batch_size, seq_len_max-1, num_vq, audio_codebook_vocab]
            # shift_labels = labels[:, 1:, 0].contiguous() # [batch_size, seq_len_max-1, num_vq]

            # compute loss
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1).to(shift_logits.device),
            )
            if dummy:
                loss = loss * 0  # 避免带来无效的梯度

        else:
            loss = None

        if generate:
            input_embeds_varlen = []
            for idx, embeds_ in enumerate(input_embeds):
                input_embeds_varlen.append(
                    embeds_[0 : input_embeds_seqlen[idx]].detach()
                )
            return {
                "loss": loss,
                "all_tts_condition_varlen": all_tts_condition_varlen,
                "gpt_spk_emb_last_hidden_states_varlen": gpt_spk_emb_last_hidden_states_varlen,
                "input_embeds_varlen": input_embeds_varlen,
            }

        return loss


if __name__ == "__main__":
    import random
    import torch
    from .processor import MelSpectrogramFeatures

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        "/home/jeeves/xubokai/ChatTTS/asset/tokenizer"
    )

    config = Config()
    model = LLMConditionalChatTTS(config)

    # model.decoder.gpt.layers[0].self_attn.q_proj.weight.dtype # torch.float32
    # model.decoder.gpt.layers[0].self_attn.q_proj.weight[0][0] # tensor(-0.0299, grad_fn=<SelectBackward0>)

    device = "cuda:0"

    processor = MelSpectrogramFeatures()  # processor could run on cpu

    model.decoder.gpt.layers[0].self_attn.q_proj.weight.dtype
    model.load_state_dict("/home/jeeves/xubokai/ChatTTS/asset", strict=False)
    model.to(torch.bfloat16)
    model.to(device)

    # after load state dict
    # model.decoder.gpt.layers[0].self_attn.q_proj.weight[0][0] tensor(-0.0014, grad_fn=<SelectBackward0>)

    # for param in model.vocos.parameters():
    #     param.require_grad = False

    for param in model.dvae.parameters():
        param.require_grad = False

    # 只训练 model.gpt 和model.llm_projection

    sequence_batch_size = 8  # LLM训练的序列数
    target_audio_batch_size = (
        20  # 比如说 `sequence_batch_size` 中一共有 `target_audio_batch_size` 个音频
    )

    seq_lens_for_sequence = [
        random.randint(10, 100) for _ in range(sequence_batch_size)
    ]
    seq_lens_for_audio_sequence = [
        random.randint(10, 100) for _ in range(target_audio_batch_size)
    ]

    llm_dim = 4096
    input_for_tts = {
        "input_ids": [
            torch.randint(0, 10001, (num,), dtype=torch.int32).to(device)
            for num in seq_lens_for_audio_sequence
        ],
        "target_audio_features": [
            processor(torch.randn(100, 100 * num, dtype=torch.float32))
            .to(torch.bfloat16)
            .to(device)
            for num in seq_lens_for_audio_sequence
        ],
        "lm_last_hidden_states": [
            torch.randn(num + 3, llm_dim, dtype=torch.bfloat16).to(device)
            for num in seq_lens_for_audio_sequence
        ],
    }

    # 从llm_last_hidden_states中取出所有的 模型应该说话的位置 的last_hidden_state 用slice的方式，这样就clone了？好像也没clone，要显式的用clone().

    with torch.cuda.amp.autocast():
        tts_loss = model(**input_for_tts)
        print("tts_loss", tts_loss)

    # 如果想要tts模型梯度累积，而LLM等待TTS模型梯度累积
    # 每个Tensor应该在外部流程中被clone一下再传入，不要和之前的计算图连同，否则loss.backward将触发整个系统的反向传播和参数更新。外部流程拿到这个loss后，可以loss.backward，然后被clone的llm hidden state会收到梯度。然后我们需要把这个梯度给加到连接原始计算图的对应hidden state里。
