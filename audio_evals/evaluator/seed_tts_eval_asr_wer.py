from typing import Dict
import os
from audio_evals.evaluator.base import Evaluator
import zhconv
from jiwer import compute_measures
from zhon.hanzi import punctuation
import re
import string
import unicodedata
import logging

logger = logging.getLogger(__name__)

punctuation_all = punctuation + string.punctuation

# zhon.hanzi.punctuation + string.punctuation 只覆盖中文标点和 ASCII 标点，
# 其余文种的标点会漏网并粘在词上（"नमस्ते।" vs "नमस्ते"），整词被判错。
# 这里统一替换成空格而不是删掉，避免 "текст―текст" 这类被粘成一个词。
extra_punctuation = (
    "\u0964\u0965"  # । ॥ 天城文 danda / 双 danda（hi）
    "\u060c\u061b\u061e\u061f\u066a\u066b\u066c\u066d\u06d4"  # ، ؛ ؞ ؟ ٪ ٫ ٬ ٭ ۔ 阿拉伯标点（ar）
    "\u0387\u037e"  # · ; 希腊语分隔号 / 希腊语问号（el）
    "\u00ab\u00bb\u2039\u203a"  # « » ‹ › 法式/西里尔引号（fr, uk, ru, el）
    "\u201a"  # ‚ 低位单引号（cs, de）
    "\u2010\u2011\u2012\u2015\u2212"  # ‐ ‑ ‒ ― − zhon 未覆盖的连接号/破折号
    "\u2024\u2025"  # ․ ‥ 点导符
    "\u00a1\u00bf"  # ¡ ¿ 倒置标点（es）
    "\u2032\u2033"  # ′ ″ 撇号/双撇号
    "\u200b"  # 零宽空格，当空格处理
)

# 弯撇号先归一成 ASCII 撇号。zhon 已包含 ’，而下面循环会跳过 "'"，
# 直接删会造成不对称：参考文本 "don’t" -> "dont"，ASR 输出 "don't" -> "don't"，整词判错。
apostrophe_variants = "\u2018\u2019\u201b\u02bc\u02bb\u00b4"

# 必须直接删除（不能换成空格，否则会把词切开）的字符：
# 阿语 harakat 短元音符号（ASR 基本不输出）、tatweel 拉伸符、零宽/方向控制符、软连字符。
invisible_and_diacritics = re.compile(
    "["
    "\u064b-\u065f"  # 阿语 harakat
    "\u0670"  # 上标 alef
    "\u06d6-\u06ed"  # 古兰经诵读符号
    "\u0640"  # tatweel
    "\u00ad"  # 软连字符
    "\u200c-\u200f\u202a-\u202e\u2066-\u2069"  # ZWNJ/ZWJ 与方向控制符
    "\u2060\ufeff"  # word joiner / BOM
    "]"
)


def normalize_text(text: str) -> str:
    """标点/变音符归一，在按 punctuation_all 删标点之前跑。"""
    # NFC：把天城文 nukta 的两种写法（क़ vs क + ़）等价序列归一。
    text = unicodedata.normalize("NFC", text)
    text = invisible_and_diacritics.sub("", text)
    for x in apostrophe_variants:
        text = text.replace(x, "'")
    for x in extra_punctuation:
        text = text.replace(x, " ")
    return text


def process_one(hypo, truth, lang):
    raw_truth = truth
    raw_hypo = hypo

    truth = normalize_text(truth)
    hypo = normalize_text(hypo)

    for x in punctuation_all:
        if x == "'":
            continue
        truth = truth.replace(x, "")
        hypo = hypo.replace(x, "")

    truth = " ".join(truth.split())
    hypo = " ".join(hypo.split())

    # yue: cantonese, th: thai
    if lang in ["zh", "ja", "yue", "th", "ko"]:
        truth = " ".join([x for x in truth if x.strip()])
        hypo = " ".join([x for x in hypo if x.strip()])
    # Word-based languages (WER)
    else:
        truth = truth.lower()
        hypo = hypo.lower()

    try:
        measures = compute_measures(truth, hypo)
        wer = measures["wer"]
    except Exception as e:
        logger.error(f"Error computing measures: {e}. truth: '{truth}', hypo: '{hypo}'")
        raise e

    return wer


class SeedTTSEvalASRWER(Evaluator):
    def __init__(self, model_name, prompt_name, lang):
        from audio_evals.registry import registry

        self.prompt = registry.get_prompt(prompt_name)
        self.model = registry.get_model(model_name)
        self.lang = lang

    def _eval(self, pred, label, **kwargs) -> Dict[str, any]:
        pred = str(pred)
        label_text = kwargs["text"]
        assert os.path.exists(pred), "must be a valid audio file, but got {}".format(
            pred
        )

        real_prompt = self.prompt.load(WavPath=pred)

        # Pass language to model for non-Chinese languages or if specified
        # Whisper model expects language in generate_kwargs
        inf_kwargs = {}
        if self.lang != "zh":
            inf_kwargs["generate_kwargs"] = {
                "language": kwargs.get("language", self.lang)
            }

        transcription = self.model.inference(real_prompt, **inf_kwargs)

        if self.lang == "zh" or kwargs.get("language") == "chinese":
            transcription = zhconv.convert(transcription, "zh-cn")

        measure_name = "cer" if self.lang in ["zh", "ja", "yue", "th", "ko"] else "wer"
        res = {
            f"{measure_name}%": process_one(transcription, label_text, self.lang) * 100
        }
        res.update(
            {
                "transcription": transcription,
                "label_text": label_text,
                "pred": pred,
                "ref": label,
            }
        )
        return res
