import argparse
import json
import select
import sys
import tempfile

import soundfile as sf
import torch
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
from qwen_omni_utils import process_mm_info


device = "cuda"


def load_model(path, **kwargs):
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        path,
        torch_dtype="auto",
        device_map="auto",
        attn_implementation="flash_attention_2",
        **kwargs,
    )

    processor = Qwen3OmniMoeProcessor.from_pretrained(path)
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
    parser.add_argument(
        "--speaker",
        type=str,
        default="Ethan",
        help="Speaker name for speech generation",
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
            print(prompt[anchor + 2 :])

            # Set whether to use audio in video
            USE_AUDIO_IN_VIDEO = True

            text = processor.apply_chat_template(
                conversation, add_generation_prompt=True, tokenize=False
            )
            audios, images, videos = process_mm_info(
                conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO
            )
            inputs = processor(
                text=text,
                audio=audios,
                images=images,
                videos=videos,
                return_tensors="pt",
                padding=True,
                use_audio_in_video=USE_AUDIO_IN_VIDEO,
            )
            inputs = inputs.to(model.device).to(model.dtype)

            # Inference: Generation of the output text and audio
            if config.speech:
                text_ids, audio = model.generate(
                    **inputs,
                    speaker=config.speaker,
                    thinker_return_dict_in_generate=True,
                    use_audio_in_video=USE_AUDIO_IN_VIDEO,
                )
                text = processor.batch_decode(
                    text_ids.sequences[:, inputs["input_ids"].shape[1] :],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                if audio is not None:
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                        sf.write(
                            f.name,
                            audio.reshape(-1).detach().cpu().numpy(),
                            samplerate=24000,
                        )
                        retry = 3
                        while retry:
                            retry -= 1
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
                    retry = 3
                    while retry:
                        retry -= 1
                        print(prefix + json.dumps({"text": text[0]}), flush=True)
                        rlist, _, _ = select.select([sys.stdin], [], [], 1)
                        if rlist:
                            finish = sys.stdin.readline().strip()
                            if finish == "{}close".format(prefix):
                                break
                        print("not found close signal, will emit again", flush=True)
            else:
                text_ids, _ = model.generate(
                    **inputs,
                    use_audio_in_video=USE_AUDIO_IN_VIDEO,
                    return_audio=False,
                    thinker_return_dict_in_generate=True,
                )
                text = processor.batch_decode(
                    text_ids.sequences[:, inputs["input_ids"].shape[1] :],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                retry = 3
                while retry:
                    retry -= 1
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
