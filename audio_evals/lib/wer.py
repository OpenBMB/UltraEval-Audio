import editdistance as ed
import zhconv

from audio_evals.lib.evaluate_tokenizer import EvaluationTokenizer
from audio_evals.lib.text_normalization.basic import BasicTextNormalizer
from audio_evals.lib.text_normalization.cn_tn import TextNorm
from audio_evals.lib.text_normalization.en import EnglishTextNormalizer

english_normalizer = EnglishTextNormalizer()
chinese_normalizer = TextNorm(
    to_banjiao=False,
    to_upper=False,
    to_lower=False,
    remove_fillers=False,
    remove_erhua=False,
    check_chars=False,
    remove_space=False,
    cc_mode="",
)
basic_normalizer = BasicTextNormalizer()


def compute_wer(refs, hyps, language="en"):
    distance = 0
    ref_length = 0
    tokenizer = EvaluationTokenizer(
        tokenizer_type="none",
        lowercase=True,
        punctuation_removal=False,
        character_tokenization=False,
    )
    for i in range(len(refs)):
        ref = refs[i]
        pred = hyps[i]

        if language in ["en"]:
            ref = english_normalizer(ref)
            pred = english_normalizer(pred)
        elif language in ["zh"]:
            ref = chinese_normalizer(ref)
            pred = chinese_normalizer(pred)
        elif language in ["yue"]:
            ref = zhconv.convert(ref, "zh-cn")
            pred = zhconv.convert(pred, "zh-cn")
        else:
            ref = basic_normalizer(ref)
            pred = basic_normalizer(pred)

        ref_items = tokenizer.tokenize(ref).split()
        pred_items = tokenizer.tokenize(pred).split()

        if language in ["zh", "yue"]:
            ref_items = [x for x in "".join(ref_items)]
            pred_items = [x for x in "".join(pred_items)]
        if len(refs) > 1 and i == 0:
            print(f"ref: {ref}")
            print(f"pred: {pred}")
            print(f"ref_items:\n{ref_items}\n{len(ref_items)}\n{ref_items[0]}")
            print(f"pred_items:\n{pred_items}\n{len(ref_items)}\n{ref_items[0]}")
        distance += ed.eval(ref_items, pred_items)
        ref_length += len(ref_items)
    return distance / ref_length
