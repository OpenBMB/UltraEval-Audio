import argparse
import json
import select
import sys
import tempfile
import os

import soundfile as sf
import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer


device = "auto"


def load_model(model_id, **kwargs):
    model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, device_map=device, **kwargs
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    return model.to("cuda"), tokenizer, processor


def process_audio_content(content):
    """Process audio content from conversation"""
    if content["type"] == "audio":
        if "path" in content:
            # Load audio from file path
            return content["path"]
        elif "url" in content:
            # For URL, we would need to download first
            # For now, just return the URL
            return content["url"]
        elif "audio" in content:
            # For numpy array, save to temp file
            import numpy as np

            audio_data = np.array(content["audio"])
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                sf.write(f.name, audio_data, samplerate=16000)
                return f.name
        elif "value" in content:
            # Handle the case where audio path is in "value" field
            if isinstance(content["value"], str):
                return content["value"]
            elif isinstance(content["value"], dict):
                if "path" in content["value"]:
                    return content["value"]["path"]
                elif "url" in content["value"]:
                    return content["value"]["url"]
                elif "audio" in content["value"]:
                    import numpy as np

                    audio_data = np.array(content["value"]["audio"])
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                        sf.write(f.name, audio_data, samplerate=16000)
                        return f.name
    return None


def process_mm_info(conversation):
    """Extract multimodal information from conversation"""
    audios = []

    for message in conversation:
        if "content" in message:
            contents = message["content"]
            if isinstance(contents, list):
                for content in contents:
                    audio_path = process_audio_content(content)
                    if audio_path:
                        audios.append(audio_path)
            else:
                # Single content
                audio_path = process_audio_content(contents)
                if audio_path:
                    audios.append(audio_path)

    return audios


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", type=str, required=True, help="Path to checkpoint file or model ID"
    )
    config = parser.parse_args()

    model, tokenizer, processor = load_model(config.path)
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
            # Inference: Generation of the output text
            with torch.no_grad():
                model_inputs = processor.apply_chat_template(
                    conversation,
                    tokenize=True,
                    add_generation_prompt=True,
                    add_special_tokens=True,
                    return_dict=True,
                )
                model_inputs = {k: v.to(model.device) for k, v in model_inputs.items()}
                generation = model.generate(**model_inputs)
                output = tokenizer.batch_decode(generation, skip_special_tokens=True)

                # Extract the generated response (remove the input prompt)
                # Find where the assistant response starts
                full_text = output[0]
                # Simple approach: find the last occurrence of "assistant" and take everything after it
                if "assistant" in full_text:
                    response = full_text.split("assistant")[-1].strip()
                else:
                    response = full_text

                retry = 3
                while retry:
                    retry -= 1
                    print(prefix + json.dumps({"text": response}), flush=True)
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
