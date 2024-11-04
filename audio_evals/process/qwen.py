import re

from audio_evals.process.base import Process


class QwenAudioASRExtract(Process):
    PUNCS = "!,.?;:"

    def __init__(self, lang: str):
        self.lang = lang

    def __call__(self, answer: str) -> str:
        gt = re.sub(r"<\|.*?\|>", " ", answer)
        gt = re.sub(rf"\s+", r" ", gt)  # 将文本中的连续空格替换为单个空格
        gt = re.sub(f" ?([{self.PUNCS}])", r"\1", gt)
        gt = gt.lstrip(" ")
        if self.lang == "zh":
            gt = re.sub(rf"\s+", r"", gt)
        return gt
