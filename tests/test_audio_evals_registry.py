import logging

import pytest

from audio_evals.eval_task import EvalTask
from audio_evals.recorder import Recorder
from audio_evals.registry import registry

# 配置根日志记录器
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)


def test_registry_model():
    model = registry.get_model("gpt4o")
    print(model.inference("how are you"))


def test_prompt():
    prompt = registry.get_prompt("asr")
    model = registry.get_model("qwen-audio-offline")
    real_prompt = prompt.load(
        a="/Users/a1/Downloads/语音转文字/嘉德罗斯/嘉德罗斯_12.wav"
    )
    print(model.inference(real_prompt))


def test_evaluator():
    e = registry.get_evaluator("em")
    assert e("0", 0)["match"]
    assert e(0, "0")["match"]
    assert e(1, "0")["match"] == 0

    e = registry.get_evaluator("cer")
    print(
        e(
            "买一张万能卡也有不少好处带着这张卡你可以进入南非的一些公园或全部的国家公园",
            "买一张万能卡（Wild Card）也有不少好处。带着这张卡，你可以进入南非的一些公园或全部的国家公园。",
        )
    )

    e = registry.get_evaluator("wer")
    print(e("It is good", "it is good"))


def test_agg():
    a = registry.get_agg("acc")
    assert a([{"match": 0}])["acc"] == 0
    assert a([{"match": 1}])["acc"] == 1
    assert a([])["acc"] == 0
    with pytest.raises(Exception):
        a([{"count": 1}])


def test_task():
    task_cfg = registry.get_eval_task("alei_asr")

    t = EvalTask(
        dataset=registry.get_dataset("KeSpeech"),
        prompt=registry.get_prompt("KeSpeech"),
        predictor=registry.get_model(task_cfg.model),
        evaluator=registry.get_evaluator(task_cfg.evaluator),
        post_process=[registry.get_process(item) for item in task_cfg.post_process],
        agg=registry.get_agg(task_cfg.agg),
        recorder=Recorder("log/KeSpeech.jsonl"),
    )
    res = t.run()
    print(res)
