import builtins

import pytest

from audio_evals import device as device_mod
from audio_evals.device import DEVICE_ENV_VAR, cuda_available, resolve_device


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.delenv(DEVICE_ENV_VAR, raising=False)


@pytest.fixture
def no_cuda(monkeypatch):
    monkeypatch.setattr(device_mod, "cuda_available", lambda: False)


@pytest.fixture
def with_cuda(monkeypatch):
    monkeypatch.setattr(device_mod, "cuda_available", lambda: True)


@pytest.fixture
def no_torch(monkeypatch):
    monkeypatch.setattr(device_mod, "cuda_available", lambda: None)


@pytest.mark.parametrize("requested", [None, "", "auto"])
def test_auto_falls_back_to_cpu_without_cuda(no_cuda, requested):
    assert resolve_device(requested) == "cpu"


@pytest.mark.parametrize("requested", [None, "", "auto"])
def test_auto_prefers_cuda_when_available(with_cuda, requested):
    assert resolve_device(requested) == "cuda:0"


def test_auto_without_torch_is_cpu(no_torch):
    assert resolve_device() == "cpu"


def test_auto_respects_custom_default_cuda(with_cuda):
    assert resolve_device(None, default_cuda="cuda:1") == "cuda:1"


def test_explicit_request_is_kept(with_cuda):
    assert resolve_device("cpu") == "cpu"
    assert resolve_device(" cuda:1 ") == "cuda:1"


def test_explicit_config_request_is_not_probed(no_cuda):
    # A device written in a registry config is passed through unchanged; the
    # isolated worker owns the final fallback because it has its own torch.
    assert resolve_device("cuda:0") == "cuda:0"


def test_env_var_overrides_requested_device(with_cuda, monkeypatch):
    monkeypatch.setenv(DEVICE_ENV_VAR, "cpu")
    assert resolve_device("cuda:0") == "cpu"


def test_env_var_cuda_is_kept_when_available(with_cuda, monkeypatch):
    monkeypatch.setenv(DEVICE_ENV_VAR, "cuda:1")
    assert resolve_device("cpu") == "cuda:1"


def test_env_var_cuda_without_cuda_raises_clear_error(no_cuda, monkeypatch):
    monkeypatch.setenv(DEVICE_ENV_VAR, "cuda:0")
    with pytest.raises(RuntimeError) as excinfo:
        resolve_device()
    message = str(excinfo.value)
    assert DEVICE_ENV_VAR in message
    assert "--device cpu" in message


def test_env_var_cuda_without_torch_is_passed_through(no_torch, monkeypatch):
    monkeypatch.setenv(DEVICE_ENV_VAR, "cuda:0")
    assert resolve_device() == "cuda:0"


@pytest.mark.parametrize("value", ["", "   ", "auto"])
def test_env_var_auto_values_mean_auto(no_cuda, monkeypatch, value):
    monkeypatch.setenv(DEVICE_ENV_VAR, value)
    assert resolve_device() == "cpu"


def test_cuda_available_follows_torch(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert cuda_available() is False
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert cuda_available() is True


def test_cuda_available_is_none_without_torch(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("no torch")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert cuda_available() is None


def test_cli_device_reaches_isolated_adapter_command(monkeypatch):
    # The adapter of an isolated model forwards the user's device choice to the
    # worker command line, overriding the registry default.
    module = pytest.importorskip("audio_evals.models.asr.fun_asr_nano")
    monkeypatch.setenv(DEVICE_ENV_VAR, "cpu")
    model = object.__new__(module.FunASRNano)
    module.FunASRNano.__init__.__wrapped__(
        model, path="FunAudioLLM/Fun-ASR-Nano-2512", device="cuda:0"
    )
    assert model.command_args["device"] == "cpu"
