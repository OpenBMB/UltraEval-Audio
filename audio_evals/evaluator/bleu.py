import sacrebleu

from audio_evals.evaluator.base import Evaluator


class BLEU(Evaluator):
    def __init__(self, lang: str = "13a"):
        self.lang = "13a"
        if lang == "zh":
            self.lang = "zh"
        elif lang == "ja":
            self.lang = "ja-mecab"

    def _eval(self, pred: str, label: str, **kwargs):
        res = sacrebleu.corpus_bleu([pred], [[label]], tokenize=self.lang)
        return {"bleu": res.score}
