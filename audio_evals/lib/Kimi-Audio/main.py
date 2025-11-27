import argparse
import json
import select
import sys
import tempfile
import logging
import time
from kimia_infer.api.kimia import KimiAudio
import soundfile as sf
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        required=False,
        default="moonshotai/Kimi-Audio-7B-Instruct",
        help="Path or HF repo for Kimi-Audio model",
    )
    parser.add_argument(
        "--speech",
        action="store_true",
        default=False,
        help="Whether to use speech output",
    )
    config = parser.parse_args()

    start_time = time.time()
    model = KimiAudio(model_path=config.model_path, load_detokenizer=True)
    end_time = time.time()
    logger.info(f"Model loading took {end_time - start_time:.2f} seconds")
    logger.info(f"Using Kimi-Audio model: {config.model_path}")

    sampling_params = {
        "audio_temperature": 0.8,
        "audio_top_k": 10,
        "text_temperature": 0.0,
        "text_top_k": 5,
        "audio_repetition_penalty": 1.0,
        "audio_repetition_window_size": 64,
        "text_repetition_penalty": 1.1,
        "text_repetition_window_size": 16,
        "max_new_tokens": -1,
    }

    while True:
        try:
            prompt = input()
            anchor = prompt.find("->")
            if anchor == -1:
                print(
                    f"Error: Invalid conversation format, must contain ->, but {prompt}",
                    flush=True,
                )
                continue
            prefix = prompt[:anchor].strip() + "->"
            x = json.loads(prompt[anchor + 2 :])

            # 兼容 MegaTTS3 的 prompt 格式，假定 x 是 PromptStruct
            # 只支持 output_type="both"，返回音频文件路径
            messages = x["messages"] if "messages" in x else x
            output_type = "both" if config.speech else "text"
            # 推理
            wav, text = model.generate(
                messages, **sampling_params, output_type=output_type
            )

            if config.speech:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    sf.write(
                        f.name,
                        (
                            wav.detach().cpu().view(-1).numpy()
                            if torch.is_tensor(wav)
                            else wav
                        ),
                        24000,
                    )
                    retry = 3
                    while retry:
                        print(
                            f"{prefix}{json.dumps({'text': text, 'audio': f.name}, ensure_ascii=False)}",
                            flush=True,
                        )
                        rlist, _, _ = select.select([sys.stdin], [], [], 1)
                        if rlist:
                            finish = sys.stdin.readline().strip()
                            if finish == f"{prefix}close":
                                break
                        print("not found close signal, will emit again", flush=True)
                        retry -= 1
            else:
                retry = 3
                while retry:
                    print(
                        f"{prefix}{json.dumps({'text': text}, ensure_ascii=False)}",
                        flush=True,
                    )
                    rlist, _, _ = select.select([sys.stdin], [], [], 1)
                    if rlist:
                        finish = sys.stdin.readline().strip()
                        if finish == f"{prefix}close":
                            break
                    print("not found close signal, will emit again", flush=True)
                    retry -= 1
        except Exception as e:
            import traceback

            traceback.print_exc()
            print(f"Error: {str(e)}", flush=True)
