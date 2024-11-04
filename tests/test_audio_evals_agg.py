import logging

import pytest

from audio_evals.registry import registry

# 配置根日志记录器
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)


def test_mean():
    for item in [[{"wer": 0.2, "text": ""}, {"wer": 0.1, "text": "1"}]]:
        a = registry.get_agg("mean")
        print(a(item))


def test_coco():
    e = registry.get_agg("coco")
    x = [
        {
            "pred": "A melodious chime is composed mostly of ascending scales.",
            "ref": [
                "A melodious chime is composed mostly of ascending scales.",
                "A set of three tones echo and then repeat",
            ],
        }
    ]
    print(e(x))
    x = [
        {
            "pred": "A melodious chime is composed mostly of ascending scales.",
            "ref": "A melodious chime is composed mostly of ascending scales.",
        }
    ]
    print(e(x))
