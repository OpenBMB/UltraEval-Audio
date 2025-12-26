import argparse
import select
import sys
import tempfile

import soundfile as sf
from sparktts.models.audio_tokenizer import BiCodecTokenizer
import torch
import logging

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", type=str, required=True, help="Path to checkpoint file"
    )
    config = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    logger.info(f"Loading tokenizer from {config.path}")
    tokenizer = BiCodecTokenizer(
        model_dir=config.path,
        device=device,
    )
    print(f"Model loaded from checkpoint: {config.path}", flush=True)

    while True:
        try:
            prompt = input()
            if not prompt.strip():
                continue

            # 解析 uuid 前缀
            anchor = prompt.find("->")
            if anchor == -1:
                print(
                    "Error: Invalid format, must contain ->, got: {}".format(prompt),
                    flush=True,
                )
                continue

            prefix = prompt[: anchor + 2]  # 包含 "->"
            audio_path = prompt[anchor + 2 :]

            global_tokens, semantic_tokens = tokenizer.tokenize(audio_path)
            wav_rec = tokenizer.detokenize(global_tokens.squeeze(0), semantic_tokens)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                sf.write(f.name, wav_rec, 16000)

                # 发送结果并等待 close 信号确认
                retry = 3
                while retry:
                    retry -= 1
                    print(f"{prefix}{f.name}", flush=True)
                    rlist, _, _ = select.select([sys.stdin], [], [], 1)
                    if rlist:
                        finish = sys.stdin.readline().strip()
                        if finish == f"{prefix}close":
                            break
                        logger.info(
                            f"not found close signal, got: {finish}, will emit again"
                        )
        except EOFError:
            # stdin 关闭，正常退出
            break
        except Exception as e:
            print(f"Error:{e}", flush=True)
