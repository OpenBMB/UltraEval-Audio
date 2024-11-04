import pytest

from audio_evals.process.qwen import QwenAudioASRExtract
from audio_evals.registry import registry


def test_qwen_audio_asr_extract():
    ex = QwenAudioASRExtract("zh")
    for s in [
        """买一张万能卡也有不少好处带着这张卡你可以进入南非的一些公园或全部的国家公园<|endoftext|>""",
    ]:
        print(ex(s))


def test_text_normalization():
    zh_normalizer = registry.get_process("zh_text_normalizer")
    text = "你好，世界！"
    print(zh_normalizer(text))

    en_normalizer = registry.get_process("en_text_normalizer")
    text = "Hello, world!"
    print(en_normalizer(text))

    normalizer = registry.get_process("text_normalizer")
    text = "你好，世界！"
    print(normalizer(text))
