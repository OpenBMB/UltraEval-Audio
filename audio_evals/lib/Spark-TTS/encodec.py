import argparse
import tempfile

import soundfile as sf
from sparktts.models.audio_tokenizer import BiCodecTokenizer
import torch
import logging

logging.basicConfig(level=logging.INFO)
device = "cuda" if torch.cuda.is_available() else "cpu"

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
    logger.info(f"successfully loaded tokenizer")

    while True:
        try:
            prompt = input()
            global_tokens, semantic_tokens = tokenizer.tokenize(prompt)
            wav_rec = tokenizer.detokenize(global_tokens.squeeze(0), semantic_tokens)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                sf.write(f.name, wav_rec, 16000)
                print("Result:" + f.name)
        except Exception as e:
            print("Error:{}".format(e))
