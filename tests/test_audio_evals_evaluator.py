import pandas as pd
import pytest
import sacrebleu

from audio_evals.agg.base import WER
from audio_evals.evaluator.bleu import BLEU
from audio_evals.registry import registry


def test_qwen_audio_asr_extract():

    e = BLEU("zh")
    for s in [("""壮哉我中华""", "壮哉我中华")]:
        print(e(*s))


def test_wer():
    e = registry.get_evaluator("wer-ignore-case")
    for s0, s1 in [
        (
            "MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL",
            "mister quilther is the apostle of the middle classes and we are glad to welcome his gospel<|endoftext|>",
        ),
    ]:
        print(e(s1, s0))


def test_coco():
    e = registry.get_evaluator("coco")
    for s0, s1 in [
        (
            "A melodious chime is composed mostly of ascending scales.",
            ["A melodious chime is composed mostly of ascending scales."],
            # "A set of three tones echo and then repeat",],
        ),
    ]:
        print(e(s0, s1))


def test_tmp():
    f_name = "/Users/a1/project/AudioEvals/log/2024-07-22_06-38-10-qwen-audio-pretrain-offline-librispeech-dev-clean.jsonl"
    df = pd.read_json(f_name, lines=True)
    df = df[df["type"] == "inference"]
    p = registry.get_process("qwen_pretrain_asr_tractor")
    e = registry.get_agg("wer-common")
    quiz = pd.read_json(
        "/Users/a1/project/AudioEvals/raw_data/librispeech/dev_clean.jsonl", lines=True
    )

    data = []
    for pred, gt in zip(df["data"].tolist(), quiz["gt"].tolist()):
        pred = p(pred["content"]).strip()
        data.append({"pred": pred, "ref": gt})
    res = e(data)["wer"]
    print(res)


def test_tmp3():
    f_name = "/Users/a1/project/AudioEvals/log/2024-07-23_09-22-25-qwen-audio-pretrain-offline-fleurs-zh.jsonl"
    raw_df = pd.read_json(f_name, lines=True)
    df = raw_df[raw_df["type"] == "inference"]
    quiz = raw_df[raw_df["type"] == "eval"]

    e = registry.get_agg("wer-zh")

    data = []
    for response, gt in zip(df["data"].tolist(), quiz["data"].tolist()):
        response = response["content"]
        gt = gt["ref"]
        data.append({"pred": response, "ref": gt})

    res = e(data)["wer"]
    print(res)


def test_tmp():
    f_name = "/Users/a1/project/AudioEvals/log/2024-07-22_06-38-10-qwen-audio-pretrain-offline-librispeech-dev-clean.jsonl"
    df = pd.read_json(f_name, lines=True)
    df = df[df["type"] == "inference"]
    p = registry.get_process("qwen_pretrain_asr_tractor")
    e = registry.get_agg("wer-ignore-case")
    quiz = pd.read_json(
        "/Users/a1/project/AudioEvals/raw_data/librispeech/dev_clean.jsonl", lines=True
    )

    data = []
    for pred, gt in zip(df["data"].tolist(), quiz["gt"].tolist()):
        pred = p(pred["content"]).strip()
        data.append({"pred": pred, "ref": gt})
    res = e(data)["wer"]
    print(res)


def test_tmp2():
    f_name = "/Users/a1/project/AudioEvals/log/2024-07-28_02-06-36-qwen-audio-pretrain-offline-mls_german.jsonl"
    df = pd.read_json(f_name, lines=True)
    df = df[df["type"] == "eval"]
    # p = registry.get_process('qwen_pretrain_asr_tractor_zh')
    e = registry.get_agg("wer-ignore-case")
    data = df["data"].tolist()
    print(e(data))
    # print(sum(res)/len(res))
