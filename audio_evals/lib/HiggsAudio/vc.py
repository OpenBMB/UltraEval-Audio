import argparse
import json
import logging
import select
import sys
import tempfile
import copy
import re
import os

import torch
import soundfile as sf
import langid
import jieba
from transformers import AutoConfig, AutoTokenizer

from boson_multimodal.serve.serve_engine import (
    HiggsAudioServeEngine,
    HiggsAudioResponse,
)
from boson_multimodal.data_types import Message, ChatMLSample, AudioContent, TextContent
from boson_multimodal.model.higgs_audio import HiggsAudioConfig, HiggsAudioModel
from boson_multimodal.data_collator.higgs_audio_collator import HiggsAudioSampleCollator
from boson_multimodal.audio_processing.higgs_audio_tokenizer import (
    load_higgs_audio_tokenizer,
)
from boson_multimodal.dataset.chatml_dataset import (
    ChatMLDatasetSample,
    prepare_chatml_sample,
)
from boson_multimodal.model.higgs_audio.utils import revert_delay_pattern
from transformers.cache_utils import StaticCache
from typing import List, Optional
from dataclasses import asdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


AUDIO_PLACEHOLDER_TOKEN = "<|__AUDIO_PLACEHOLDER__|>"
MULTISPEAKER_DEFAULT_SYSTEM_MESSAGE = """You are an AI assistant designed to convert text into speech.
If the user's message includes a [SPEAKER*] tag, do not read out the tag and generate speech for the following text, using the specified voice.
If no speaker tag is present, select a suitable voice on your own."""


def normalize_chinese_punctuation(text):
    """Convert Chinese (full-width) punctuation marks to English (half-width) equivalents."""
    chinese_to_english_punct = {
        "，": ", ",
        "。": ".",
        "：": ":",
        "；": ";",
        "？": "?",
        "！": "!",
        "（": "(",
        "）": ")",
        "【": "[",
        "】": "]",
        "《": "<",
        "》": ">",
        """: '"', """: '"',
        "'": "'",
        "'": "'",
        "、": ",",
        "—": "-",
        "…": "...",
        "·": ".",
        "「": '"',
        "」": '"',
        "『": '"',
        "』": '"',
    }
    for zh_punct, en_punct in chinese_to_english_punct.items():
        text = text.replace(zh_punct, en_punct)
    return text


def prepare_chunk_text(
    text,
    chunk_method: Optional[str] = None,
    chunk_max_word_num: int = 100,
    chunk_max_num_turns: int = 1,
):
    """Chunk the text into smaller pieces."""
    if chunk_method is None:
        return [text]
    elif chunk_method == "speaker":
        lines = text.split("\n")
        speaker_chunks = []
        speaker_utterance = ""
        for line in lines:
            line = line.strip()
            if line.startswith("[SPEAKER") or line.startswith("<|speaker_id_start|>"):
                if speaker_utterance:
                    speaker_chunks.append(speaker_utterance.strip())
                speaker_utterance = line
            else:
                if speaker_utterance:
                    speaker_utterance += "\n" + line
                else:
                    speaker_utterance = line
        if speaker_utterance:
            speaker_chunks.append(speaker_utterance.strip())
        if chunk_max_num_turns > 1:
            merged_chunks = []
            for i in range(0, len(speaker_chunks), chunk_max_num_turns):
                merged_chunk = "\n".join(speaker_chunks[i : i + chunk_max_num_turns])
                merged_chunks.append(merged_chunk)
            return merged_chunks
        return speaker_chunks
    elif chunk_method == "word":
        language = langid.classify(text)[0]
        paragraphs = text.split("\n\n")
        chunks = []
        for idx, paragraph in enumerate(paragraphs):
            if language == "zh":
                words = list(jieba.cut(paragraph, cut_all=False))
                for i in range(0, len(words), chunk_max_word_num):
                    chunk = "".join(words[i : i + chunk_max_word_num])
                    chunks.append(chunk)
            else:
                words = paragraph.split(" ")
                for i in range(0, len(words), chunk_max_word_num):
                    chunk = " ".join(words[i : i + chunk_max_word_num])
                    chunks.append(chunk)
            chunks[-1] += "\n\n"
        return chunks
    else:
        raise ValueError(f"Unknown chunk method: {chunk_method}")


def _build_system_message_with_audio_prompt(system_message):
    contents = []
    while AUDIO_PLACEHOLDER_TOKEN in system_message:
        loc = system_message.find(AUDIO_PLACEHOLDER_TOKEN)
        contents.append(TextContent(system_message[:loc]))
        contents.append(AudioContent(audio_url=""))
        system_message = system_message[loc + len(AUDIO_PLACEHOLDER_TOKEN) :]
    if len(system_message) > 0:
        contents.append(TextContent(system_message))
    ret = Message(role="system", content=contents)
    return ret


class HiggsAudioModelClient:
    def __init__(
        self,
        model_path,
        audio_tokenizer,
        device=None,
        device_id=None,
        max_new_tokens=2048,
        kv_cache_lengths: List[int] = [1024, 4096, 8192],
        use_static_kv_cache=False,
    ):
        if device_id is not None:
            device = f"cuda:{device_id}"
            self._device = device
        else:
            if device is not None:
                self._device = device
            else:
                if torch.cuda.is_available():
                    self._device = "cuda:0"
                elif torch.backends.mps.is_available():
                    self._device = "mps"
                else:
                    self._device = "cpu"

        logger.info(f"Using device: {self._device}")
        if isinstance(audio_tokenizer, str):
            audio_tokenizer_device = "cpu" if self._device == "mps" else self._device
            self._audio_tokenizer = load_higgs_audio_tokenizer(
                audio_tokenizer, device=audio_tokenizer_device
            )
        else:
            self._audio_tokenizer = audio_tokenizer

        self._model = HiggsAudioModel.from_pretrained(
            model_path,
            device_map=self._device,
            torch_dtype=torch.bfloat16,
        )
        self._model.eval()
        self._kv_cache_lengths = kv_cache_lengths
        self._use_static_kv_cache = use_static_kv_cache

        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        self._config = AutoConfig.from_pretrained(model_path)
        self._max_new_tokens = max_new_tokens
        self._collator = HiggsAudioSampleCollator(
            whisper_processor=None,
            audio_in_token_id=self._config.audio_in_token_idx,
            audio_out_token_id=self._config.audio_out_token_idx,
            audio_stream_bos_id=self._config.audio_stream_bos_id,
            audio_stream_eos_id=self._config.audio_stream_eos_id,
            encode_whisper_embed=self._config.encode_whisper_embed,
            pad_token_id=self._config.pad_token_id,
            return_audio_in_tokens=self._config.encode_audio_in_tokens,
            use_delay_pattern=self._config.use_delay_pattern,
            round_to=1,
            audio_num_codebooks=self._config.audio_num_codebooks,
        )
        self.kv_caches = None
        if use_static_kv_cache:
            self._init_static_kv_cache()

    def _init_static_kv_cache(self):
        cache_config = copy.deepcopy(self._model.config.text_config)
        cache_config.num_hidden_layers = (
            self._model.config.text_config.num_hidden_layers
        )
        if self._model.config.audio_dual_ffn_layers:
            cache_config.num_hidden_layers += len(
                self._model.config.audio_dual_ffn_layers
            )
        self.kv_caches = {
            length: StaticCache(
                config=cache_config,
                max_batch_size=1,
                max_cache_len=length,
                device=self._model.device,
                dtype=self._model.dtype,
            )
            for length in sorted(self._kv_cache_lengths)
        }
        if "cuda" in self._device:
            logger.info(f"Capturing CUDA graphs for each KV cache length")
            self._model.capture_model(self.kv_caches.values())

    def _prepare_kv_caches(self):
        for kv_cache in self.kv_caches.values():
            kv_cache.reset()

    @torch.inference_mode()
    def generate(
        self,
        messages,
        audio_ids,
        chunked_text,
        generation_chunk_buffer_size,
        temperature=1.0,
        top_k=50,
        top_p=0.95,
        ras_win_len=7,
        ras_win_max_num_repeat=2,
        seed=123,
        *args,
        **kwargs,
    ):
        if ras_win_len is not None and ras_win_len <= 0:
            ras_win_len = None
        sr = 24000
        audio_out_ids_l = []
        generated_audio_ids = []
        generation_messages = []
        for idx, chunk_text in enumerate(chunked_text):
            generation_messages.append(Message(role="user", content=chunk_text))
            chatml_sample = ChatMLSample(messages=messages + generation_messages)
            input_tokens, _, _, _ = prepare_chatml_sample(
                chatml_sample, self._tokenizer
            )
            postfix = self._tokenizer.encode(
                "<|start_header_id|>assistant<|end_header_id|>\n\n",
                add_special_tokens=False,
            )
            input_tokens.extend(postfix)

            logger.info(f"========= Chunk {idx} Input =========")
            logger.info(self._tokenizer.decode(input_tokens))
            context_audio_ids = audio_ids + generated_audio_ids

            curr_sample = ChatMLDatasetSample(
                input_ids=torch.LongTensor(input_tokens),
                label_ids=None,
                audio_ids_concat=(
                    torch.concat([ele.cpu() for ele in context_audio_ids], dim=1)
                    if context_audio_ids
                    else None
                ),
                audio_ids_start=(
                    torch.cumsum(
                        torch.tensor(
                            [0] + [ele.shape[1] for ele in context_audio_ids],
                            dtype=torch.long,
                        ),
                        dim=0,
                    )
                    if context_audio_ids
                    else None
                ),
                audio_waveforms_concat=None,
                audio_waveforms_start=None,
                audio_sample_rate=None,
                audio_speaker_indices=None,
            )

            batch_data = self._collator([curr_sample])
            batch = asdict(batch_data)
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.contiguous().to(self._device)

            if self._use_static_kv_cache:
                self._prepare_kv_caches()

            outputs = self._model.generate(
                **batch,
                max_new_tokens=self._max_new_tokens,
                use_cache=True,
                do_sample=True,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                past_key_values_buckets=self.kv_caches,
                ras_win_len=ras_win_len,
                ras_win_max_num_repeat=ras_win_max_num_repeat,
                stop_strings=["<|end_of_text|>", "<|eot_id|>"],
                tokenizer=self._tokenizer,
                seed=seed,
            )

            step_audio_out_ids_l = []
            for ele in outputs[1]:
                audio_out_ids = ele
                if self._config.use_delay_pattern:
                    audio_out_ids = revert_delay_pattern(audio_out_ids)
                step_audio_out_ids_l.append(
                    audio_out_ids.clip(0, self._audio_tokenizer.codebook_size - 1)[
                        :, 1:-1
                    ]
                )
            audio_out_ids = torch.concat(step_audio_out_ids_l, dim=1)
            audio_out_ids_l.append(audio_out_ids)
            generated_audio_ids.append(audio_out_ids)

            generation_messages.append(
                Message(role="assistant", content=AudioContent(audio_url=""))
            )
            if (
                generation_chunk_buffer_size is not None
                and len(generated_audio_ids) > generation_chunk_buffer_size
            ):
                generated_audio_ids = generated_audio_ids[
                    -generation_chunk_buffer_size:
                ]
                generation_messages = generation_messages[
                    (-2 * generation_chunk_buffer_size) :
                ]

        logger.info(f"========= Final Text output =========")
        logger.info(self._tokenizer.decode(outputs[0][0]))
        concat_audio_out_ids = torch.concat(audio_out_ids_l, dim=1)

        if concat_audio_out_ids.device.type == "mps":
            concat_audio_out_ids_cpu = concat_audio_out_ids.detach().cpu()
        else:
            concat_audio_out_ids_cpu = concat_audio_out_ids

        concat_wv = self._audio_tokenizer.decode(concat_audio_out_ids_cpu.unsqueeze(0))[
            0, 0
        ]
        text_result = self._tokenizer.decode(outputs[0][0])
        return concat_wv, sr, text_result


def prepare_generation_context(
    scene_prompt,
    ref_audio,
    ref_audio_text,
    ref_audio_in_system_message,
    audio_tokenizer,
    speaker_tags,
):
    """Prepare the context for generation."""
    system_message = None
    messages = []
    audio_ids = []
    if ref_audio is not None:
        num_speakers = len(ref_audio.split(","))
        speaker_info_l = ref_audio.split(",")
        ref_audio_text_l = (
            ref_audio_text.split(",") if ref_audio_text else [None] * num_speakers
        )

        voice_profile = None
        if any(
            [
                speaker_info.startswith("profile:")
                for speaker_info in ref_audio.split(",")
            ]
        ):
            ref_audio_in_system_message = True
        if ref_audio_in_system_message:
            speaker_desc = []
            for spk_id, character_name in enumerate(speaker_info_l):
                if character_name.startswith("profile:"):
                    if voice_profile is None:
                        import yaml

                        voice_profile_path = os.path.join(
                            os.path.dirname(__file__), "voice_prompts/profile.yaml"
                        )
                        with open(voice_profile_path, "r", encoding="utf-8") as f:
                            voice_profile = yaml.safe_load(f)
                    character_desc = voice_profile["profiles"][
                        character_name[len("profile:") :].strip()
                    ]
                    speaker_desc.append(f"SPEAKER{spk_id}: {character_desc}")
                else:
                    speaker_desc.append(f"SPEAKER{spk_id}: {AUDIO_PLACEHOLDER_TOKEN}")
            if scene_prompt:
                system_message = (
                    "Generate audio following instruction."
                    "\n\n"
                    f"<|scene_desc_start|>\n{scene_prompt}\n\n"
                    + "\n".join(speaker_desc)
                    + "\n<|scene_desc_end|>"
                )
            else:
                system_message = (
                    "Generate audio following instruction.\n\n"
                    + f"<|scene_desc_start|>\n"
                    + "\n".join(speaker_desc)
                    + "\n<|scene_desc_end|>"
                )
            system_message = _build_system_message_with_audio_prompt(system_message)
        else:
            if scene_prompt:
                system_message = Message(
                    role="system",
                    content=f"Generate audio following instruction.\n\n<|scene_desc_start|>\n{scene_prompt}\n<|scene_desc_end|>",
                )
        voice_profile = None
        for spk_id, character_name in enumerate(ref_audio.split(",")):
            if not character_name.startswith("profile:"):
                # Check if it's a file path (contains / or \ or ends with common audio extensions)
                is_file_path = (
                    "/" in character_name
                    or "\\" in character_name
                    or character_name.endswith(
                        (".wav", ".mp3", ".flac", ".ogg", ".m4a")
                    )
                )

                if is_file_path:
                    # Use the provided file path directly
                    prompt_audio_path = character_name
                    assert os.path.exists(
                        prompt_audio_path
                    ), f"Audio file {prompt_audio_path} does not exist."

                    # Use provided text or empty string
                    if ref_audio_text_l[spk_id]:
                        prompt_text = ref_audio_text_l[spk_id].strip()
                    else:
                        prompt_text = ""
                        logger.warning(
                            f"No prompt_text provided for {prompt_audio_path}, using empty string."
                        )
                else:
                    # Use predefined voice prompts (original behavior)
                    prompt_audio_path = os.path.join(
                        os.path.dirname(__file__),
                        "voice_prompts",
                        f"{character_name}.wav",
                    )
                    prompt_text_path = os.path.join(
                        os.path.dirname(__file__),
                        "voice_prompts",
                        f"{character_name}.txt",
                    )
                    assert os.path.exists(
                        prompt_audio_path
                    ), f"Voice prompt audio file {prompt_audio_path} does not exist."
                    assert os.path.exists(
                        prompt_text_path
                    ), f"Voice prompt text file {prompt_text_path} does not exist."
                    with open(prompt_text_path, "r", encoding="utf-8") as f:
                        prompt_text = f.read().strip()

                audio_tokens = audio_tokenizer.encode(prompt_audio_path)
                audio_ids.append(audio_tokens)

                if not ref_audio_in_system_message:
                    messages.append(
                        Message(
                            role="user",
                            content=(
                                f"[SPEAKER{spk_id}] {prompt_text}"
                                if num_speakers > 1
                                else prompt_text
                            ),
                        )
                    )
                    messages.append(
                        Message(
                            role="assistant",
                            content=AudioContent(audio_url=prompt_audio_path),
                        )
                    )
    else:
        if len(speaker_tags) > 1:
            speaker_desc_l = []
            for idx, tag in enumerate(speaker_tags):
                if idx % 2 == 0:
                    speaker_desc = f"feminine"
                else:
                    speaker_desc = f"masculine"
                speaker_desc_l.append(f"{tag}: {speaker_desc}")
            speaker_desc = "\n".join(speaker_desc_l)
            scene_desc_l = []
            if scene_prompt:
                scene_desc_l.append(scene_prompt)
            scene_desc_l.append(speaker_desc)
            scene_desc = "\n\n".join(scene_desc_l)
            system_message = Message(
                role="system",
                content=f"{MULTISPEAKER_DEFAULT_SYSTEM_MESSAGE}\n\n<|scene_desc_start|>\n{scene_desc}\n<|scene_desc_end|>",
            )
        else:
            system_message_l = ["Generate audio following instruction."]
            if scene_prompt:
                system_message_l.append(
                    f"<|scene_desc_start|>\n{scene_prompt}\n<|scene_desc_end|>"
                )
            system_message = Message(
                role="system", content="\n\n".join(system_message_l)
            )
    if system_message:
        messages.insert(0, system_message)
    return messages, audio_ids


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", type=str, required=True, help="Path to HiggsAudio model"
    )
    parser.add_argument(
        "--audio_tokenizer",
        type=str,
        default="bosonai/higgs-audio-v2-tokenizer",
        help="Audio tokenizer path",
    )
    parser.add_argument("--ras_win_len", type=int, default=7, help="RAS window length")
    parser.add_argument(
        "--ras_win_max_num_repeat", type=int, default=2, help="RAS max repeat"
    )
    parser.add_argument(
        "--use_static_kv_cache", type=int, default=1, help="Use static KV cache"
    )
    parser.add_argument("--device", type=str, default="auto", help="Device")
    args = parser.parse_args()

    device_id = None
    if args.device == "auto":
        if torch.cuda.is_available():
            device_id = 0
            device = "cuda:0"
        elif torch.backends.mps.is_available():
            device_id = None
            device = "mps"
        else:
            device_id = None
            device = "cpu"
    elif args.device.startswith("cuda"):
        device_id = int(args.device.split(":")[-1]) if ":" in args.device else 0
        device = args.device
    else:
        device = args.device

    logger.info(f"Using device: {device}")
    logger.info(f"Loading HiggsAudio model from {args.path}")

    # For MPS, use CPU for audio tokenizer
    audio_tokenizer_device = "cpu" if device == "mps" else device
    audio_tokenizer = load_higgs_audio_tokenizer(
        args.audio_tokenizer, device=audio_tokenizer_device
    )

    # Disable static KV cache on MPS
    use_static_kv_cache = args.use_static_kv_cache and device != "mps"

    model_client = HiggsAudioModelClient(
        model_path=args.path,
        audio_tokenizer=audio_tokenizer,
        device=device,
        device_id=device_id,
        max_new_tokens=2048,  # Default value, will be overridden per request
        use_static_kv_cache=use_static_kv_cache,
    )

    logger.info("HiggsAudio model successfully loaded")

    pattern = re.compile(r"\[(SPEAKER\d+)\]")

    while True:
        try:
            prompt = input()
            anchor = prompt.find("->")
            if anchor == -1:
                print(
                    f"Error: Invalid conversation format, must contain '->', but got {prompt}",
                    flush=True,
                )
                continue

            prefix = prompt[:anchor].strip() + "->"
            x = json.loads(prompt[anchor + 2 :])

            transcript = x.get("text", "")
            scene_prompt = x.get(
                "scene_prompt",
                "third_party/higgs-audio/examples/scene_prompts/quiet_indoor.txt",
            )
            ref_audio = x.get("prompt_audio", None)
            ref_audio_text = x.get("prompt_text", None)
            ref_audio_in_system_message = x.get("ref_audio_in_system_message", False)
            chunk_method = x.get("chunk_method", None)
            chunk_max_word_num = x.get("chunk_max_word_num", 200)
            chunk_max_num_turns = x.get("chunk_max_num_turns", 1)
            generation_chunk_buffer_size = x.get("generation_chunk_buffer_size", None)
            seed = x.get("seed", None)

            # Get generation params from input, use defaults if not provided
            max_new_tokens = x.get("max_new_tokens", 2048)
            temperature = x.get("temperature", 1.0)
            top_k = x.get("top_k", 50)
            top_p = x.get("top_p", 0.95)
            ras_win_len = x.get("ras_win_len", args.ras_win_len)
            ras_win_max_num_repeat = x.get(
                "ras_win_max_num_repeat", args.ras_win_max_num_repeat
            )

            speaker_tags = sorted(set(pattern.findall(transcript)))

            # Normalize text
            transcript = normalize_chinese_punctuation(transcript)
            transcript = transcript.replace("(", " ").replace(")", " ")
            transcript = transcript.replace("°F", " degrees Fahrenheit").replace(
                "°C", " degrees Celsius"
            )

            # Replace special tags
            for tag, replacement in [
                ("[laugh]", "<SE>[Laughter]</SE>"),
                ("[humming start]", "<SE_s>[Humming]</SE_s>"),
                ("[humming end]", "<SE_e>[Humming]</SE_e>"),
                ("[music start]", "<SE_s>[Music]</SE_s>"),
                ("[music end]", "<SE_e>[Music]</SE_e>"),
                ("[music]", "<SE>[Music]</SE>"),
                ("[sing start]", "<SE_s>[Singing]</SE_s>"),
                ("[sing end]", "<SE_e>[Singing]</SE_e>"),
                ("[applause]", "<SE>[Applause]</SE>"),
                ("[cheering]", "<SE>[Cheering]</SE>"),
                ("[cough]", "<SE>[Cough]</SE>"),
            ]:
                transcript = transcript.replace(tag, replacement)

            lines = transcript.split("\n")
            transcript = "\n".join(
                [" ".join(line.split()) for line in lines if line.strip()]
            )
            transcript = transcript.strip()

            if not any(
                [
                    transcript.endswith(c)
                    for c in [".", "!", "?", ",", ";", '"', "'", "</SE_e>", "</SE>"]
                ]
            ):
                transcript += "."

            messages, audio_ids = prepare_generation_context(
                scene_prompt=scene_prompt,
                ref_audio=ref_audio,
                ref_audio_text=ref_audio_text,
                ref_audio_in_system_message=ref_audio_in_system_message,
                audio_tokenizer=audio_tokenizer,
                speaker_tags=speaker_tags,
            )

            chunked_text = prepare_chunk_text(
                transcript,
                chunk_method=chunk_method,
                chunk_max_word_num=chunk_max_word_num,
                chunk_max_num_turns=chunk_max_num_turns,
            )

            logger.info("Generating audio...")

            # Update model client's max_new_tokens if provided
            if max_new_tokens != model_client._max_new_tokens:
                model_client._max_new_tokens = max_new_tokens

            concat_wv, sr, text_output = model_client.generate(
                messages=messages,
                audio_ids=audio_ids,
                chunked_text=chunked_text,
                generation_chunk_buffer_size=generation_chunk_buffer_size,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                ras_win_len=ras_win_len,
                ras_win_max_num_repeat=ras_win_max_num_repeat,
                seed=seed,
            )

            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                sf.write(f.name, concat_wv, samplerate=sr)

            retry = 3
            while retry:
                retry -= 1
                print(f"{prefix}{f.name}", flush=True)
                rlist, _, _ = select.select([sys.stdin], [], [], 1)
                if rlist:
                    finish = sys.stdin.readline().strip()
                    if finish == f"{prefix}close":
                        break
                print("not found close signal, will emit again", flush=True)

        except Exception as e:
            logger.error(f"Error processing request: {str(e)}")
            import traceback

            traceback.print_exc()
            print(f"Error: {str(e)}", flush=True)
