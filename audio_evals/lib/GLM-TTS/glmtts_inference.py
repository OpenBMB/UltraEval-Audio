# Copyright (c) 2025 Zhipu AI Inc (authors: CogAudio Group Members)
# Authors: Jiayan Cui, Zhihan Yang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Modified for UltraEval-Audio: parameterized all hardcoded paths

import logging
import os
import torch

from cosyvoice.cli.frontend import TTSFrontEnd, SpeechTokenizer, TextFrontEnd
from utils import tts_model_util, yaml_util, seed_util
from transformers import AutoTokenizer, LlamaForCausalLM
from llm.glmtts import GLMTTS
from utils.audio import mel_spectrogram
from functools import partial

# --- Global Constants ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LLM_SEQ_INP_LEN = 750
TOKEN_RATE = 25

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_special_token_ids(tokenize_fn):
    """Get special token IDs based on the tokenizer name."""
    _special_token_ids = {
        "ats": "<|audio_0|>",
        "ate": "<|audio_32767|>",
        "boa": "<|begin_of_audio|>",
        "eoa": "<|user|>",
        "pad": "<|endoftext|>",
    }

    special_token_ids = {}
    endoftext_id = tokenize_fn("<|endoftext|>")[0]

    for k, v in _special_token_ids.items():
        __ids = tokenize_fn(v)
        if len(__ids) != 1:
            raise AssertionError(
                f"Token '{k}' ({v}) encoded to multiple tokens: {__ids}"
            )
        if __ids[0] < endoftext_id:
            raise AssertionError(
                f"Token '{k}' ({v}) ID {__ids[0]} < endoftext ID {endoftext_id}"
            )
        special_token_ids[k] = __ids[0]

    return special_token_ids


def _assert_shape_and_get_len(token):
    assert token.ndim == 2 and token.shape[0] == 1
    token_len = torch.tensor([token.shape[1]], dtype=torch.int32).to(token.device)
    return token_len


def load_frontends(
    speech_tokenizer, ckpt_dir, repo_dir, sample_rate=24000, use_phoneme=False
):
    """
    Load frontend components.

    Args:
        speech_tokenizer: SpeechTokenizer instance
        ckpt_dir: Path to checkpoint directory (e.g., "ckpt")
        repo_dir: Path to GLM-TTS repo directory
        sample_rate: Audio sample rate
        use_phoneme: Whether to use phoneme mode
    """
    if sample_rate == 32000:
        feat_extractor = partial(
            mel_spectrogram,
            sampling_rate=sample_rate,
            hop_size=640,
            n_fft=2560,
            num_mels=80,
            win_size=2560,
            fmin=0,
            fmax=8000,
            center=False,
        )
        logger.info("Configured for 32kHz frontend.")
    elif sample_rate == 24000:
        feat_extractor = partial(
            mel_spectrogram,
            sampling_rate=sample_rate,
            hop_size=480,
            n_fft=1920,
            num_mels=80,
            win_size=1920,
            fmin=0,
            fmax=8000,
            center=False,
        )
        logger.info("Configured for 24kHz frontend.")
    else:
        raise ValueError(f"Unsupported sampling_rate: {sample_rate}")

    # Load tokenizer from ckpt_dir
    tokenizer_path = os.path.join(ckpt_dir, "vq32k-phoneme-tokenizer")
    glm_tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, trust_remote_code=True
    )
    tokenize_fn = lambda text: glm_tokenizer.encode(text)

    # frontend_dir is in repo_dir
    frontend_dir = os.path.join(repo_dir, "frontend")

    frontend = TTSFrontEnd(
        tokenize_fn,
        speech_tokenizer,
        feat_extractor,
        os.path.join(frontend_dir, "campplus.onnx"),
        os.path.join(frontend_dir, "spk2info.pt"),
        DEVICE,
    )
    text_frontend = TextFrontEnd(use_phoneme)
    return frontend, text_frontend


def load_models(
    ckpt_dir, repo_dir="./third_party/GLM-TTS", use_phoneme=False, sample_rate=24000
):
    """
    Load all model components.

    Args:
        ckpt_dir: Path to checkpoint directory containing model weights
        repo_dir: Path to GLM-TTS repo directory (default: ./third_party/GLM-TTS)
        use_phoneme: Whether to use phoneme mode
        sample_rate: Audio sample rate
    """
    # Load Speech Tokenizer
    speech_tokenizer_path = os.path.join(ckpt_dir, "speech_tokenizer")
    _model, _feature_extractor = yaml_util.load_speech_tokenizer(speech_tokenizer_path)
    speech_tokenizer = SpeechTokenizer(_model, _feature_extractor)

    # Load Frontends
    frontend, text_frontend = load_frontends(
        speech_tokenizer,
        ckpt_dir,
        repo_dir,
        sample_rate=sample_rate,
        use_phoneme=use_phoneme,
    )

    # Load LLM
    llama_path = os.path.join(ckpt_dir, "llm")
    llm = GLMTTS(
        llama_cfg_path=os.path.join(llama_path, "config.json"),
        mode="PRETRAIN",
        spk_prompt_dict_path=os.path.join(repo_dir, "configs/spk_prompt_dict.yaml"),
        lora_adapter_config=os.path.join(
            repo_dir, "configs/lora_adapter_configV3.1.json"
        ),
    )
    llm.llama = LlamaForCausalLM.from_pretrained(llama_path).to(DEVICE)
    llm.llama_embedding = llm.llama.model.embed_tokens

    special_token_ids = get_special_token_ids(frontend.tokenize_fn)
    llm.set_runtime_vars(special_token_ids=special_token_ids)

    # Load Flow
    flow_dir = os.path.join(ckpt_dir, "flow")
    flow = yaml_util.load_flow_model(
        os.path.join(flow_dir, "flow.pt"), os.path.join(flow_dir, "config.yaml"), DEVICE
    )
    token2wav = tts_model_util.Token2Wav(flow, sample_rate=sample_rate, device=DEVICE)

    return frontend, text_frontend, speech_tokenizer, llm, token2wav


def local_llm_forward(
    llm,
    prompt_text_token,
    tts_text_token,
    prompt_speech_token,
    beam_size=1,
    sampling=25,
    sample_method="ras",
):
    """Single LLM forward pass."""
    prompt_text_token_len = _assert_shape_and_get_len(prompt_text_token)
    tts_text_token_len = _assert_shape_and_get_len(tts_text_token)
    prompt_speech_token_len = _assert_shape_and_get_len(prompt_speech_token)

    tts_speech_token = llm.inference(
        text=tts_text_token,
        text_len=tts_text_token_len,
        prompt_text=prompt_text_token,
        prompt_text_len=prompt_text_token_len,
        prompt_speech_token=prompt_speech_token,
        prompt_speech_token_len=prompt_speech_token_len,
        beam_size=beam_size,
        sampling=sampling,
        sample_method=sample_method,
        spk=None,
    )
    return tts_speech_token[0].tolist()


def local_flow_forward(flow, token_list, prompt_speech_tokens, speech_feat, embedding):
    """Single Flow forward pass."""
    wav, full_mel = flow.token2wav_with_cache(
        token_list,
        prompt_token=prompt_speech_tokens,
        prompt_feat=speech_feat,
        embedding=embedding,
    )
    return wav.detach().cpu(), full_mel


def get_cached_prompt(cache, synth_text_token, device=DEVICE):
    """
    Constructs prompt tokens from the cache.
    Prunes the cache if the sequence length exceeds MAX_LLM_SEQ_INP_LEN.
    """
    cache_text = cache["cache_text"]
    cache_text_token = cache["cache_text_token"]
    cache_speech_token = cache["cache_speech_token"]

    def __len_cache_text_token():
        return sum(map(lambda x: x.shape[1], cache_text_token))

    def __len_cache_speech_token():
        return sum(map(len, cache_speech_token))

    text_len = __len_cache_text_token()
    ta_ratio = __len_cache_speech_token() / (text_len if text_len > 0 else 1.0)

    __len_synth_text_token = synth_text_token.shape[1]
    __len_synth_audi_token_estim = int(ta_ratio * __len_synth_text_token)

    # Prune cache if too long
    while (
        __len_cache_speech_token() + __len_synth_audi_token_estim > MAX_LLM_SEQ_INP_LEN
    ):
        if len(cache_speech_token) <= 1:
            break
        cache_text.pop(1)
        cache_text_token.pop(1)
        cache_speech_token.pop(1)

    # Construct Text Prompt
    prompt_text_token_from_cache = []
    for a_token in cache_text_token:
        prompt_text_token_from_cache.extend(a_token.squeeze().tolist())
    prompt_text_token = torch.tensor([prompt_text_token_from_cache]).to(device)

    # Construct Speech Prompt
    speech_tokens = []
    for a_cache_speech_token in cache_speech_token:
        speech_tokens.extend(a_cache_speech_token)
    llm_speech_token = torch.tensor([speech_tokens], dtype=torch.int32).to(device)

    return prompt_text_token, llm_speech_token


def generate_long(
    frontend,
    text_frontend,
    llm,
    flow,
    text_info,
    cache,
    device,
    embedding,
    seed=0,
    sample_method="ras",
    flow_prompt_token=None,
    speech_feat=None,
    use_phoneme=False,
):
    """Generate speech for potentially long text."""
    outputs = []
    full_mels = []
    output_token_list = []
    uttid = text_info[0]
    syn_text = text_info[1]

    text_tn_dict = {
        "uttid": uttid,
        "syn_text": syn_text,
        "syn_text_tn": [],
        "syn_text_phoneme": [],
    }

    short_text_list = text_frontend.split_by_len(syn_text)

    for _, tts_text in enumerate(short_text_list):
        seed_util.set_seed(seed)
        tts_text_tn = text_frontend.text_normalize(tts_text)
        text_tn_dict["syn_text_tn"].append(tts_text_tn)

        if use_phoneme:
            tts_text_tn = text_frontend.g2p_infer(tts_text_tn)
            text_tn_dict["syn_text_phoneme"].append(tts_text_tn)

        tts_text_token = frontend._extract_text_token(tts_text_tn)

        cache_text = cache["cache_text"]
        cache_text_token = cache["cache_text_token"]
        cache_speech_token = cache["cache_speech_token"]

        # Determine Prompts
        if cache["use_cache"] and len(cache_text_token) > 1:
            prompt_text_token, prompt_speech_token = get_cached_prompt(
                cache, tts_text_token, device
            )
        else:
            prompt_text_token = cache_text_token[0].to(device)
            prompt_speech_token = torch.tensor(
                [cache_speech_token[0]], dtype=torch.int32
            ).to(device)

        # LLM Inference
        token_list_res = local_llm_forward(
            llm=llm,
            prompt_text_token=prompt_text_token,
            tts_text_token=tts_text_token,
            prompt_speech_token=prompt_speech_token,
            sample_method=sample_method,
        )
        output_token_list.extend(token_list_res)

        # Flow Inference
        output, full_mel = local_flow_forward(
            flow=flow,
            token_list=token_list_res,
            prompt_speech_tokens=flow_prompt_token,
            speech_feat=speech_feat,
            embedding=embedding,
        )

        # Update Cache
        if cache is not None:
            cache_text.append(tts_text_tn)
            cache_text_token.append(tts_text_token)
            cache_speech_token.append(token_list_res)

        outputs.append(output)
        if full_mel is not None:
            full_mels.append(full_mel)

    tts_speech = torch.concat(outputs, dim=1)
    tts_mel = torch.concat(full_mels, dim=-1) if full_mels else None

    return tts_speech, tts_mel, output_token_list, text_tn_dict
