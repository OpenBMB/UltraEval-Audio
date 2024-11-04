import re

from audio_evals.lib.text_normalization.basic import BasicTextNormalizer
from audio_evals.lib.text_normalization.cn_tn import TextNorm
from audio_evals.lib.text_normalization.en import EnglishTextNormalizer
from audio_evals.process.base import Process


class TextNormalization(Process):

    def __init__(self, lang: str = ""):
        if lang == "en":
            self.normalizer = EnglishTextNormalizer()
        elif lang == "zh":
            self.normalizer = TextNorm(
                to_banjiao=False,
                to_upper=False,
                to_lower=False,
                remove_fillers=False,
                remove_erhua=False,
                check_chars=False,
                remove_space=False,
                cc_mode="",
            )
        else:
            self.normalizer = BasicTextNormalizer()

    def __call__(self, answer: str) -> str:
        return self.normalizer(answer)
