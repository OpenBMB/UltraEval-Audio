import logging

from audio_evals.registry import registry

# 配置根日志记录器
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)


def test_huggingface_dataset():
    a = registry.get_dataset("KeSpeech-hf")
    b = a.load()
    b = list(b)
    with open(b[0]["audio"]["path"], "rb") as f:
        content = f.read()
    print(content)
    print(a)
