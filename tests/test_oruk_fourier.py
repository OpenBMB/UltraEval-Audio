import importlib
import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_registry_declares_reproducible_oruk_fourier_2b_model():
    registry_path = ROOT / "registry/model/oruk-fourier-2b.yaml"
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    spec = registry.get("oruk-fourier-2b")

    assert spec is not None
    assert spec["class"] == "audio_evals.models.asr.oruk_fourier.OrukFourierASR"
    assert spec["args"] == {
        "path": "oruk/oruk-fourier-2b",
        "model_revision": "8c2a8871f1304aa2f634da4fb95e24438c4945b9",
        "dtype": "bfloat16",
        "device_map": "cuda:0",
        "max_new_tokens": 512,
        "language": "en",
        "env_path": "envs/oruk_fourier",
        "requirements_path": "audio_evals/lib/OrukFourier/requirements.txt",
    }


def test_adapter_builds_worker_command_from_registry_options():
    module = importlib.import_module("audio_evals.models.asr.oruk_fourier")
    model = object.__new__(module.OrukFourierASR)

    module.OrukFourierASR.__init__.__wrapped__(
        model,
        path="oruk/oruk-fourier-2b",
        model_revision="revision-abc",
        dtype="bfloat16",
        device_map="cuda:0",
        max_new_tokens=512,
        language="en",
    )

    assert model.command_args == {
        "path": "oruk/oruk-fourier-2b",
        "model_revision": "revision-abc",
        "dtype": "bfloat16",
        "device_map": "cuda:0",
        "max_new_tokens": 512,
        "language": "en",
    }


def test_adapter_request_uses_prompt_audio_and_language_precedence():
    module = importlib.import_module("audio_evals.models.asr.oruk_fourier")

    assert module.build_request(
        {"audio": "/tmp/sample.wav", "language": "English"},
        default_language="en",
        language="fr",
    ) == {"audio": "/tmp/sample.wav", "language": "English"}

    assert module.build_request(
        {"audio": "/tmp/sample.wav"},
        default_language="en",
    ) == {"audio": "/tmp/sample.wav", "language": "en"}


def test_adapter_request_rejects_missing_audio():
    module = importlib.import_module("audio_evals.models.asr.oruk_fourier")

    with pytest.raises(ValueError, match="audio"):
        module.build_request({"language": "en"})


def test_adapter_exchanges_request_and_close_signal_with_worker():
    module = importlib.import_module("audio_evals.models.asr.oruk_fourier")
    model = object.__new__(module.OrukFourierASR)
    model.default_language = "en"
    worker_code = """
import json
import sys

request_line = sys.stdin.readline().strip()
prefix, payload = request_line.split("->", 1)
request = json.loads(payload)
assert request == {"audio": "/tmp/sample.wav", "language": "en"}
print(f"{prefix}->hello world", flush=True)
assert sys.stdin.readline().strip() == f"{prefix}->close"
"""
    model.process = subprocess.Popen(
        [sys.executable, "-u", "-c", worker_code],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        assert model._inference({"audio": "/tmp/sample.wav"}) == "hello world"
        assert model.process.wait(timeout=5) == 0
    finally:
        if model.process.poll() is None:
            model.process.terminate()
            model.process.wait(timeout=5)


def _load_worker_module():
    path = ROOT / "audio_evals/lib/OrukFourier/main.py"
    spec = importlib.util.spec_from_file_location("oruk_fourier_worker", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_worker_returns_normalized_transcript(monkeypatch):
    module = _load_worker_module()
    monkeypatch.setattr(
        module, "load_audio_16k", lambda p: np.zeros(1600, dtype=np.float32)
    )

    class FakeInputs(dict):
        def to(self, device, dtype=None):
            return self

    class FakeTensor:
        def __init__(self, shape):
            self.shape = shape

        def __getitem__(self, item):
            return self

    class FakeProcessor:
        def apply_transcription_request(self, audio, language=None):
            assert len(audio) == 1
            assert language == "en"
            return FakeInputs({"input_ids": FakeTensor((1, 4))})

        def decode(self, ids, return_format="transcription_only"):
            assert return_format == "transcription_only"
            return ["  hello\nworld  "]

    class FakeModel:
        device = "cpu"
        dtype = "bfloat16"

        def generate(self, **kwargs):
            assert kwargs["max_new_tokens"] == 512
            return FakeTensor((1, 8))

    assert (
        module.transcribe(
            FakeModel(),
            FakeProcessor(),
            "/tmp/sample.wav",
            language="en",
            max_new_tokens=512,
        )
        == "hello world"
    )


def test_worker_rejects_empty_results(monkeypatch):
    module = _load_worker_module()
    monkeypatch.setattr(
        module, "load_audio_16k", lambda p: np.zeros(1600, dtype=np.float32)
    )

    class FakeInputs(dict):
        def to(self, device, dtype=None):
            return self

    class FakeTensor:
        def __init__(self, shape):
            self.shape = shape

        def __getitem__(self, item):
            return self

    class EmptyProcessor:
        def apply_transcription_request(self, audio, language=None):
            return FakeInputs({"input_ids": FakeTensor((1, 4))})

        def decode(self, ids, return_format="transcription_only"):
            return []

    class FakeModel:
        device = "cpu"
        dtype = "bfloat16"

        def generate(self, **kwargs):
            return FakeTensor((1, 8))

    with pytest.raises(RuntimeError, match="no result"):
        module.transcribe(
            FakeModel(), EmptyProcessor(), "/tmp/sample.wav", language="en"
        )
