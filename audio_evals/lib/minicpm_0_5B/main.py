import argparse
import json
import select
import sys
import time

import librosa
from transformers import AutoModel, AutoConfig, AutoProcessor, AutoTokenizer
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(ckpt_path, pt_path):
    config = AutoConfig.from_pretrained(ckpt_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(ckpt_path, config=config, trust_remote_code=True)
    model = model.to(torch.bfloat16)
    state_dict = torch.load(pt_path, map_location="cpu")
    if "module" in state_dict:
        state_dict = state_dict["module"]
    params_dict = {}
    for key in state_dict:
        if key.startswith("llm.model"):
            new_key = key.replace("llm.model.", "model.")

        elif key.startswith("apm"):
            new_key = key.replace("apm", "audio_encoder")

        elif key.startswith("audio_projection_layer"):
            new_key = key.replace("audio_projection_layer", "projection_layer")
        elif key.startswith("llm"):
            new_key = key.replace("llm.", "")
        else:
            new_key = key

        if new_key not in params_dict:
            params_dict[new_key] = state_dict[key].clone()
        else:
            params_dict[new_key] += state_dict[key]

    missing_keys, unexpected_keys = model.load_state_dict(params_dict, strict=False)
    model.eval()
    model.to(device)
    processor = AutoProcessor.from_pretrained(ckpt_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path, trust_remote_code=True)

    return model, processor, tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", type=str, required=True, help="Path to checkpoint file"
    )

    config = parser.parse_args()
    raw_path = "/data/shiqundong/model/qwen_mlp_05b/"
    model, processor, tokenizer = load_model(raw_path, config.path)
    print("Model loaded from checkpoint: {}".format(config.path))

    while True:
        try:
            prompt = input()
            anchor = prompt.find("->")
            if anchor == -1:
                print(
                    f"invalid input format. Expected 'prefix->json_input', got '{prompt}'",
                    file=sys.stderr,
                    flush=True,
                )
                continue

            prefix = prompt[:anchor].strip() + "->"
            input_json_str = prompt[anchor + 2 :].strip()
            conversation = json.loads(input_json_str)
            text, audio = conversation["text"], conversation["audio"]
            waveforms = [librosa.load(audio, sr=16000)[0]]

            model_inputs = {}
            audio_inputs = processor.feature_extractor(
                waveforms,
                sampling_rate=16000,
                return_attention_mask=True,
                padding="max_length",
                return_tensors="pt",
            )
            model_inputs["feature_lens"] = (
                audio_inputs["attention_mask"].sum(dim=1).to(device)
            )
            model_inputs["wavforms_or_feats"] = (
                audio_inputs["input_features"][
                    :, :, : model_inputs["feature_lens"].max().item()
                ]
                .to(device)
                .type(torch.bfloat16)
            )

            text_inputs = tokenizer([text] * len(waveforms), return_tensors="pt")
            model_inputs["input_ids"] = text_inputs["input_ids"].to(device)
            model_inputs["attention_mask"] = text_inputs["attention_mask"].to(device)

            res = model.generate(
                **model_inputs,
                pad_token_id=tokenizer.eos_token_id,
                max_new_tokens=128,
                num_beams=1,
                chunk_length=-1.0,
                use_interleave_embed=False,
                repetition_penalty=1.0,
            )

            generate_ids = res[:, model_inputs["input_ids"].shape[1] :]
            generated_hyps = tokenizer.batch_decode(
                generate_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            # Wait for acknowledgment
            ack_wait_start = time.time()
            while time.time() - ack_wait_start < 60:  # 60s timeout
                res_content = json.dumps(
                    {"text": generated_hyps[0]}, ensure_ascii=False
                )
                print(f"{prefix}{res_content}", flush=True)
                print(
                    f"Sent results for text: {text[:50]}... Waiting for ack...",
                    flush=True,
                )
                rlist, _, xlist = select.select([sys.stdin], [], [sys.stdin], 1.0)
                if rlist:
                    ack_signal = sys.stdin.readline().strip()
                    expected_ack = f"{prefix.strip('->')}->ok"
                    if ack_signal == expected_ack:
                        break
                    else:
                        print(
                            f"Warning: Received unexpected input while waiting for ack for {prefix}: '{ack_signal}'. Expected '{expected_ack}'",
                            file=sys.stderr,
                            flush=True,
                        )

        except Exception as e:
            import traceback

            traceback.print_exc()
            print("Error:" + str(e))
