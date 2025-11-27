# some functions are copied from https://github.com/FunAudioLLM/CosyVoice/blob/main/cosyvoice/utils/frontend_utils.py
import re
import regex
import inflect
from functools import partial
from tn.chinese.normalizer import Normalizer as ZhNormalizer
from tn.english.normalizer import Normalizer as EnNormalizer


def normal_cut_sentence(text):
    # 先处理括号内的逗号，将其替换为特殊标记
    text = re.sub(r"([（(][^）)]*)([,，])([^）)]*[）)])", r"\1&&&\3", text)
    text = re.sub(
        "([。！，？\?])([^’”])", r"\1\n\2", text
    )  # 普通断句符号且后面没有引号
    text = re.sub("(\.{6})([^’”])", r"\1\n\2", text)  # 英文省略号且后面没有引号
    text = re.sub("(\…{2})([^’”])", r"\1\n\2", text)  # 中文省略号且后面没有引号
    text = re.sub(
        "([. ,。！；？\?\.{6}\…{2}][’”])([^’”])", r"\1\n\2", text
    )  # 断句号+引号且后面没有引号
    # 处理英文句子的分隔
    text = re.sub(
        r'([.,!?])([^’”\'"])', r"\1\n\2", text
    )  # 句号、感叹号、问号后面没有引号
    text = re.sub(
        r'([.!?][’”\'"])([^’”\'"])', r"\1\n\2", text
    )  # 句号、感叹号、问号加引号后面的部分
    text = re.sub(r"([（(][^）)]*)(&&&)([^）)]*[）)])", r"\1，\3", text)
    text = [t for t in text.split("\n") if t]
    return text


def cut_sentence_with_fix_length(text: str, length: int):
    sentences = normal_cut_sentence(text)
    cur_length = 0
    res = ""
    for sentence in sentences:
        if not sentence:
            continue
        if cur_length > length or cur_length + len(sentence) > length:
            yield res
            res = ""
            cur_length = 0
        res += sentence
        cur_length += len(sentence)
    if res:
        yield res


chinese_char_pattern = re.compile(r"[\u4e00-\u9fff]+")


# whether contain chinese character
def contains_chinese(text):
    return bool(chinese_char_pattern.search(text))


# replace special symbol
def replace_corner_mark(text):
    text = text.replace("²", "平方")
    text = text.replace("³", "立方")
    text = text.replace("√", "根号")
    text = text.replace("≈", "约等于")
    text = text.replace("<", "小于")
    return text


# remove meaningless symbol
def remove_bracket(text):
    text = text.replace("（", " ").replace("）", " ")
    text = text.replace("【", " ").replace("】", " ")
    text = text.replace("`", "").replace("`", "")
    text = text.replace("——", " ")
    return text


# spell Arabic numerals
def spell_out_number(text: str, inflect_parser):
    new_text = []
    st = None
    for i, c in enumerate(text):
        if not c.isdigit():
            if st is not None:
                num_str = inflect_parser.number_to_words(text[st:i])
                new_text.append(num_str)
                st = None
            new_text.append(c)
        else:
            if st is None:
                st = i
    if st is not None and st < len(text):
        num_str = inflect_parser.number_to_words(text[st:])
        new_text.append(num_str)
    return "".join(new_text)


# split paragrah logic：
# 1. per sentence max len token_max_n, min len token_min_n, merge if last sentence len less than merge_len
# 2. cal sentence len according to lang
# 3. split sentence according to puncatation
def split_paragraph(
    text: str,
    tokenize,
    lang="zh",
    token_max_n=80,
    token_min_n=60,
    merge_len=20,
    comma_split=False,
):
    def calc_utt_length(_text: str):
        if lang == "zh":
            return len(_text)
        else:
            return len(tokenize(_text))

    def should_merge(_text: str):
        if lang == "zh":
            return len(_text) < merge_len
        else:
            return len(tokenize(_text)) < merge_len

    if lang == "zh":
        pounc = ["。", "？", "！", "；", "：", "、", ".", "?", "!", ";"]
    else:
        pounc = [".", "?", "!", ";", ":"]
    if comma_split:
        pounc.extend(["，", ","])
    st = 0
    utts = []
    for i, c in enumerate(text):
        if c in pounc:
            if len(text[st:i]) > 0:
                utts.append(text[st:i] + c)
            if i + 1 < len(text) and text[i + 1] in ['"', "”"]:
                tmp = utts.pop(-1)
                utts.append(tmp + text[i + 1])
                st = i + 2
            else:
                st = i + 1
    if len(utts) == 0:
        if lang == "zh":
            utts.append(text + "。")
        else:
            utts.append(text + ".")
    final_utts = []
    cur_utt = ""
    for utt in utts:
        if (
            calc_utt_length(cur_utt + utt) > token_max_n
            and calc_utt_length(cur_utt) > token_min_n
        ):
            final_utts.append(cur_utt)
            cur_utt = ""
        cur_utt = cur_utt + utt
    if len(cur_utt) > 0:
        if should_merge(cur_utt) and len(final_utts) != 0:
            final_utts[-1] = final_utts[-1] + cur_utt
        else:
            final_utts.append(cur_utt)

    return final_utts


# remove blank between chinese character
def replace_blank(text: str):
    out_str = []
    for i, c in enumerate(text):
        if c == " ":
            if (text[i + 1].isascii() and text[i + 1] != " ") and (
                text[i - 1].isascii() and text[i - 1] != " "
            ):
                out_str.append(c)
        else:
            out_str.append(c)
    return "".join(out_str)


def clean_markdown(md_text: str) -> str:
    # 去除代码块 ``` ```（包括多行）
    md_text = re.sub(r"```.*?```", "", md_text, flags=re.DOTALL)

    # 去除内联代码 `code`
    md_text = re.sub(r"`[^`]*`", "", md_text)

    # 去除图片语法 ![alt](url)
    md_text = re.sub(r"!\[[^\]]*\]\([^\)]+\)", "", md_text)

    # 去除链接但保留文本 [text](url) -> text
    md_text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", md_text)

    # 替换无序列表符号
    md_text = re.sub(r"^(\s*)-\s+", r"\1", md_text, flags=re.MULTILINE)

    # 去除HTML标签
    md_text = re.sub(r"<[^>]+>", "", md_text)

    # 去除标题符号（#）
    md_text = re.sub(r"^#{1,6}\s*", "", md_text, flags=re.MULTILINE)

    # 去除多余空格和空行
    md_text = re.sub(r"\n\s*\n", "\n", md_text)  # 多余空行
    md_text = md_text.strip()

    return md_text


def clean_text(text):
    # 去除 Markdown 语法
    text = clean_markdown(text)
    # 匹配并移除表情符号
    text = regex.compile(
        r"\p{Emoji_Presentation}|\p{Emoji}\uFE0F", flags=regex.UNICODE
    ).sub("", text)
    # 去除换行符
    text = text.replace("\n", " ")
    text = text.replace("\t", " ")
    text = text.replace('"', "\“")
    return text


class TextNormalizer:
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer
        self.zh_tn_model = ZhNormalizer(
            remove_erhua=False,
            full_to_half=False,
            remove_interjections=False,
            overwrite_cache=True,
        )
        self.en_tn_model = EnNormalizer()
        self.inflect_parser = inflect.engine()

    def normalize(self, text, split=False):
        # 去除 Markdown 语法，去除表情符号，去除换行符
        lang = "zh" if contains_chinese(text) else "en"
        text = clean_text(text)
        if lang == "zh":
            text = text.replace(
                "=", "等于"
            )  # 修复 ”550 + 320 等于 870 千卡。“ 被错误正则为 ”五百五十加三百二十等于八七十千卡.“
            if re.search(r"([\d$%^*_+≥≤≠×÷?=])", text):  # 避免 英文连字符被错误正则为减
                text = re.sub(
                    r"(?<=[a-zA-Z0-9])-(?=\d)", " - ", text
                )  # 修复 x-2 被正则为 x负2
                text = self.zh_tn_model.normalize(text)
            text = re.sub(
                r"(?<=[a-zA-Z0-9])-(?=\d)", " - ", text
            )  # 修复 x-2 被正则为 x负2
            text = self.zh_tn_model.normalize(text)
            text = replace_blank(text)
            text = replace_corner_mark(text)
            text = remove_bracket(text)
            text = re.sub(r"[，,]+$", "。", text)
        else:
            text = self.en_tn_model.normalize(text)
            text = spell_out_number(text, self.inflect_parser)
        if split is False:
            return text


if __name__ == "__main__":
    text_normalizer = TextNormalizer()
    text = r"""今天我们学习一元二次方程。一元二次方程的标准形式是：
ax2+bx+c=0ax^2 + bx + c = 0ax2+bx+c=0
其中，aaa、bbb 和 ccc 是常数，xxx 是变量。这个方程的解可以通过求根公式来找到。
一元二次方程的解法有几种：
  - 因式分解法：通过将方程因式分解来求解。我们首先尝试将方程表达成两个括号的形式，解决方程的解。比如，方程x2−5x+6=0x^2 - 5x + 6 = 0x2−5x+6=0可以因式分解为(x−2)(x−3)=0(x - 2)(x - 3) = 0(x−2)(x−3)=0，因此根为2和3。
  - 配方法：通过配方将方程转化为完全平方的形式，从而解出。我们通过加上或减去适当的常数来完成这一过程，使得方程可以直接写成一个完全平方的形式。
  - 求根公式：我们可以使用求根公式直接求出方程的解。这个公式适用于所有的一元二次方程，即使我们无法通过因式分解或配方法来解决时，也能使用该公式。
公式：x=−b±b2−4ac2ax = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}x=2a−b±b2−4ac​​这个公式可以帮助我们求解任何一元二次方程的根。
对于一元二次方程，我们需要了解判别式。判别式的作用是帮助我们判断方程的解的个数和性质。判别式 Δ\DeltaΔ 由下式给出：Δ=b2−4ac\Delta = b^2 - 4acΔ=b2−4ac 根据判别式的值，我们可以知道：
  - 如果 Δ>0\Delta > 0Δ>0，方程有两个不相等的实数解。这是因为判别式大于0时，根号内的值是正数，所以我们可以得到两个不同的解。
  - 如果 Δ=0\Delta = 0Δ=0，方程有一个实数解。这是因为根号内的值为零，导致两个解相等，也就是说方程有一个解。
  - 如果 Δ<0\Delta < 0Δ<0，方程没有实数解。这意味着根号内的值是负数，无法进行实数运算，因此方程没有实数解，可能有复数解。"""
    texts = [
        "这是一个公式 (a+b)³=a³+3a²b+3ab²+b³ S=(a×b)÷2",
        "这样的发展为AI仅仅作为“工具”这一观点提出了新的挑战，",
        "550 + 320 = 870千卡。",
        "解一元二次方程：3x^2+x-2=0",
        "你好啊",
    ]
    texts = [text]
    for text in texts:
        text = text_normalizer.normalize(text)
        print(text)
        for t in cut_sentence_with_fix_length(text, 15):
            print(t)
