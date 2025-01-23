import json
import tempfile
from typing import Dict
import librosa
from audio_evals.base import PromptStruct
from audio_evals.models.model import Model
import torch
from transformers import AutoModel, AutoTokenizer

# load omni model default, the default init_vision/init_audio/init_tts is True
# if load vision-only model, please set init_audio=False and init_tts=False
# if load audio-only model, please set init_vision=False


class MiniCPMoAudio(Model):
    def __init__(self, path: str, sample_params: Dict[str, any] = None):
        super().__init__(True, sample_params)
        model = AutoModel.from_pretrained(
            path,
            trust_remote_code=True,
            attn_implementation="sdpa",  # sdpa or flash_attention_2
            torch_dtype=torch.bfloat16,
            init_vision=True,
            init_audio=True,
            init_tts=True,
        )

        self.model = model.eval().cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        self.model.init_tts()
        self.model.tts.float()
        ref_audio_path = "assets/default.wav"
        self.ref_audio, _ = librosa.load(ref_audio_path, sr=16000, mono=True)

    def _inference(self, prompt: PromptStruct, **kwargs) -> str:
        msgs = []
        for content in prompt:
            if content["role"] == "user":
                msg_line = {"role": "user", "content": []}
                for line in content["contents"]:
                    if line["type"] == "text":
                        msg_line["content"].append(line["value"])
                    if line["type"] == "audio":
                        msg_line["content"].append(
                            librosa.load(line["value"], sr=16000, mono=True)[0]
                        )
                msgs.append(msg_line)
            if content["role"] == "system":
                msg_line = {"role": "system", "content": []}
                for line in content["contents"]:
                    if line["type"] == "text":
                        msg_line["content"].append(line["value"])
                msgs.append(msg_line)

        res = self.model.chat(
            msgs=msgs, tokenizer=self.tokenizer, use_tts_template=True, **kwargs
        )
        return res


class MiniCPMoSpeech(MiniCPMoAudio):
    def _inference(self, prompt: PromptStruct, **kwargs) -> str:
        sys_msg = {
            "role": "user",
            "content": [
                "Use the voice in the audio prompt to synthesize new content.",
                self.ref_audio,
                "You are a helpful assistant with the above voice style.",
            ],
        }
        audio_file = ""
        msgs = [sys_msg]
        for content in prompt:
            if content["role"] == "user":
                for line in content["contents"]:
                    if line["type"] == "audio":
                        audio_file = line["value"]
        msgs.append(
            {
                "role": "user",
                "content": [librosa.load(audio_file, sr=16000, mono=True)[0]],
            }
        )
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            res = self.model.chat(
                msgs=msgs,
                tokenizer=self.tokenizer,
                use_tts_template=True,
                generate_audio=True,
                stream=False,
                stream_input=True,
                use_tts=True,
                output_audio_path=f.name,
                **kwargs
            )
            return json.dumps({"audio": f.name, "text": res.text}, ensure_ascii=False)
