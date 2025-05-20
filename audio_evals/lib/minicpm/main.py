import argparse
import json
import select
import sys
import tempfile

import librosa
from transformers import AutoModel, AutoConfig, AutoProcessor, AutoTokenizer
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", type=str, required=True, help="Path to checkpoint file"
    )
    parser.add_argument(
        "--speech", action="store_true", default=False, help="Path to checkpoint file"
    )

    config = parser.parse_args()
    model = AutoModel.from_pretrained(
        config.path,
        trust_remote_code=True,
        attn_implementation="sdpa",  # sdpa or flash_attention_2
        torch_dtype=torch.bfloat16,
        init_vision=True,
        init_audio=True,
        init_tts=True,
    )
    model = model.eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(config.path, trust_remote_code=True)
    model.init_tts()
    model.tts.float()
    if config.speech:
        model.config.stream_input = True
        ref_audio_path = "assets/default.wav"
        ref_audio, _ = librosa.load(ref_audio_path, sr=16000, mono=True)

    print("Model loaded from checkpoint: {}".format(config.path))

    while True:
        try:
            prompt = input()
            anchor = prompt.find("->")
            if anchor == -1:
                print(
                    "Error: Invalid conversation format, must contains  ->, but {}".format(
                        prompt
                    ),
                    flush=True,
                )
                continue
            prefix = prompt[:anchor].strip() + "->"
            conversation = json.loads(prompt[anchor + 2 :])
            assert "prompt" in conversation
            print("input is:{}".format(prompt))
            prompt = conversation.pop("prompt")

            # Inference: Generation of the output text and audio
            if config.speech:
                sys_msg = {
                    "role": "user",
                    "content": [
                        "Use the voice in the audio prompt to synthesize new content.",
                        ref_audio,
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
                    res = model.chat(
                        msgs=msgs,
                        tokenizer=tokenizer,
                        use_tts_template=True,
                        generate_audio=True,
                        stream=False,
                        stream_input=True,
                        use_tts=True,
                        output_audio_path=f.name,
                        **conversation,
                    )
                    retry = 3
                    while retry:
                        retry -= 1
                        print(
                            prefix
                            + json.dumps(
                                {"text": res.text, "audio": f.name}, ensure_ascii=False
                            ),
                            flush=True,
                        )
                        rlist, _, _ = select.select([sys.stdin], [], [], 1)
                        if rlist:
                            finish = sys.stdin.readline().strip()
                            if finish == "{}close".format(prefix):
                                break
                            print("not found close signal, will emit again", flush=True)
            else:
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
                res = model.chat(
                    msgs=msgs,
                    tokenizer=tokenizer,
                    use_tts_template=True,
                    **conversation,
                )
                retry = 3
                while retry:
                    retry -= 1
                    print(
                        prefix + json.dumps({"text": res}, ensure_ascii=False),
                        flush=True,
                    )
                    rlist, _, _ = select.select([sys.stdin], [], [], 1)
                    if rlist:
                        finish = sys.stdin.readline().strip()
                        if finish == "{}close".format(prefix):
                            break
                        print("not found close signal, will emit again", flush=True)
        except Exception as e:
            import traceback

            traceback.print_exc()
            print("Error:" + str(e))
