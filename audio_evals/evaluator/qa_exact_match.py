import re
import unicodedata
from typing import Dict

from audio_evals.evaluator.base import Evaluator
import regex
import string

'''
from https://github.com/DevSinghSachan/emdr2/blob/edb8cf6701bfa4ad7c961a21cdb461f4f7a0c72f/tasks/openqa/dense_retriever/evaluation/qa_validation.py
'''

def normalize_answer(s):
    def remove_articles(text):
        return regex.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


class QAExactMatchEvaluator(Evaluator):

    def _eval(self, pred, label, **kwargs) -> Dict[str, any]:

        if isinstance(label, list):
            for item in label:
                ans = self._eval(pred, item, **kwargs)
                if ans["match"] == 1:
                    return ans
            return {"match": 0, "pred": pred, "ref": label}

        match = exact_match_score(str(pred), str(label))

        return {
            "match": 1 if match else 0,
            "pred": label if match else pred,
            "ref": label,
        }


class SimpleTokenizer(object):
    ALPHA_NUM = r'[\p{L}\p{N}\p{M}]+'
    NON_WS = r'[^\p{Z}\p{C}]'

    def __init__(self):
        """
        Args:
            annotators: None or empty set (only tokenizes).
        """
        self._regexp = regex.compile(
            '(%s)|(%s)' % (self.ALPHA_NUM, self.NON_WS),
            flags=regex.IGNORECASE + regex.UNICODE + regex.MULTILINE
        )

    def tokenize(self, text, uncased=False):
        matches = [m for m in self._regexp.finditer(text)]
        if uncased:
            tokens = [m.group().lower() for m in matches]
        else:
            tokens = [m.group() for m in matches]
        return tokens


s_tokenizer = SimpleTokenizer()


def _normalize(text):
    return unicodedata.normalize('NFD', text)


def has_answer(answers, text, tokenizer) -> bool:
    """Check if a document contains an answer string."""
    text = _normalize(text)
    text = tokenizer.tokenize(text, uncased=True)

    for answer in answers:
        answer = _normalize(answer)
        answer = tokenizer.tokenize(answer, uncased=True)
        for i in range(0, len(text) - len(answer) + 1):
            if answer == text[i: i + len(answer)]:
                return True
    return False


class QAExistMatchEvaluator(Evaluator):

    def _eval(self, pred, label, **kwargs) -> Dict[str, any]:

        if isinstance(label, str):
            label = [label]

        match = has_answer(label, str(pred), s_tokenizer)

        return {
            "match": 1 if match else 0,
            "pred": label if match else pred,
            "ref": label,
        }