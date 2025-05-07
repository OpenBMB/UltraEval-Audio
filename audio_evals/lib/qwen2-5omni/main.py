import argparse
import json
import select
import sys
import tempfile

import soundfile as sf
import torch
from transformers import Qwen2_5OmniModel, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info


device = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(path, **kwargs):
    model = Qwen2_5OmniModel.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        attn_implementation="flash_attention_2",
        **kwargs
    )

    processor = Qwen2_5OmniProcessor.from_pretrained(path)
    return model, processor


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", type=str, required=True, help="Path to checkpoint file"
    )
    parser.add_argument(
        "--speech",
        action="store_true",
        default=False,
        help="Whether to use speech output",
    )
    config = parser.parse_args()
    model, processor = load_model(config.path)
    print("Model loaded from checkpoint: {}".format(config.path), flush=True)

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
            text = processor.apply_chat_template(
                conversation, add_generation_prompt=True, tokenize=False
            )
            audios, images, videos = process_mm_info(
                conversation, use_audio_in_video=True
            )
            inputs = processor(
                text=text,
                audios=audios,
                images=images,
                videos=videos,
                return_tensors="pt",
                padding=True,
            )
            inputs = inputs.to(model.device).to(model.dtype)

            # Inference: Generation of the output text and audio
            if config.speech:
                text_ids, audio = model.generate(
                    **inputs, use_audio_in_video=True, thinker_do_sample=False
                )
                text = processor.batch_decode(
                    text_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    sf.write(
                        f.name,
                        audio.reshape(-1).detach().cpu().numpy(),
                        samplerate=24000,
                    )
                    while True:
                        print(
                            prefix + json.dumps({"text": text[0], "audio": f.name}),
                            flush=True,
                        )
                        rlist, _, _ = select.select([sys.stdin], [], [], 1)
                        if rlist:
                            finish = sys.stdin.readline().strip()
                            if finish == "{}close".format(prefix):
                                break
                        print("not found close signal, will emit again", flush=True)

            else:
                text_ids = model.generate(
                    **inputs,
                    use_audio_in_video=True,
                    return_audio=False,
                    thinker_do_sample=False
                )
                text = processor.batch_decode(
                    text_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                while True:
                    print(prefix + json.dumps({"text": text[0]}), flush=True)
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
