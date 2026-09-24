import base64
import struct
from unittest.mock import Mock

import pytest

from audio_evals.models.TTS import qwen_audio_tts_next as module
from audio_evals.models.TTS.qwen_audio_tts_next import QwenAudioTTSNext


def _wav(data: bytes, riff_size: int = None, data_size: int = None) -> bytes:
    """Build a minimal mono 24kHz WAV, optionally with bogus chunk sizes."""
    fmt = struct.pack("<4sIHHIIHH", b"fmt ", 16, 1, 1, 24000, 48000, 2, 16)
    body = fmt + struct.pack("<4sI", b"data", data_size or len(data)) + data
    return b"RIFF" + struct.pack("<I", riff_size or (len(body) + 4)) + b"WAVE" + body


@pytest.fixture
def model(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    return QwenAudioTTSNext(workspace_id="test-workspace")


@pytest.fixture
def reference(tmp_path):
    path = tmp_path / "reference.wav"
    path.write_bytes(b"reference audio")
    return str(path)


def _capture_request(monkeypatch, model, prompt, audio=b"generated audio"):
    """Run _inference against a stubbed endpoint and return the posted input."""
    response = Mock()
    response.ok = True
    response.json.return_value = {
        "output": {
            "finish_reason": "stop",
            "audio": {"data": base64.b64encode(audio).decode("ascii")},
        }
    }
    post = Mock(return_value=response)
    monkeypatch.setattr(module.requests, "post", post)
    output_path = model.inference(prompt)
    return post.call_args.kwargs["json"]["input"], output_path


# --- text_prompt composition (pure) -----------------------------------------


def test_composes_plain_tts_prompt(model):
    composed = model._compose_text_prompt("你好，世界", None, False)
    assert "你好，世界" in composed
    assert "@voice1" not in composed
    assert "朗读内容" not in composed


def test_composes_instruction_prompt(model):
    composed = model._compose_text_prompt("Good morning", "Speak cheerfully.", False)
    assert composed == "Speak cheerfully.\n朗读内容：“Good morning”"
    assert "@voice1" not in composed


def test_composes_instruction_with_reference_prompt(model):
    composed = model._compose_text_prompt("Good morning", "Speak cheerfully.", True)
    assert "@voice1" in composed
    assert "Speak cheerfully." in composed
    assert "Good morning" in composed


def test_composes_clone_prompt_without_instruction(model):
    composed = model._compose_text_prompt("你好", None, True)
    assert "@voice1" in composed
    assert "你好" in composed


def test_instruction_whitespace_is_stripped(model):
    assert model._compose_text_prompt("hi", "  Speak softly.  ", False).startswith(
        "Speak softly."
    )


# --- _inference wiring -------------------------------------------------------


def test_instruction_is_folded_not_rejected(monkeypatch, model):
    """Regression guard: `instruction` must not hit the option whitelist."""
    request_input, _ = _capture_request(
        monkeypatch,
        model,
        {"text": "Good morning", "instruction": "Speak cheerfully."},
    )
    assert "instruction" not in request_input
    assert "Speak cheerfully." in request_input["text_prompt"]
    assert "Good morning" in request_input["text_prompt"]
    assert "references" not in request_input


def test_stock_instruct_tts_vc_prompt_shape(monkeypatch, model, reference):
    """The stock `instruct-tts-vc` prompt renders text+instruction+prompt_audio."""
    request_input, _ = _capture_request(
        monkeypatch,
        model,
        {
            "text": "你好，世界",
            "instruction": "用悲伤的语气说话。",
            "prompt_audio": reference,
        },
    )
    assert "@voice1" in request_input["text_prompt"]
    assert len(request_input["references"]) == 1
    assert request_input["references"][0]["audio_data"].startswith(
        "data:audio/wav;base64,"
    )


def test_empty_instruction_is_ignored(monkeypatch, model):
    """Prompt templates render a missing column as an empty string."""
    request_input, _ = _capture_request(
        monkeypatch, model, {"text": "你好", "instruction": ""}
    )
    assert "朗读内容" not in request_input["text_prompt"]


def test_explicit_text_prompt_wins_over_instruction(monkeypatch, model):
    request_input, _ = _capture_request(
        monkeypatch,
        model,
        {"text_prompt": "verbatim prompt", "instruction": "Speak cheerfully."},
    )
    assert request_input["text_prompt"] == "verbatim prompt"


def test_unknown_options_are_still_rejected(model):
    with pytest.raises(ValueError, match="Unsupported Qwen-Audio-TTS-Next options"):
        model._inference({"text": "hi", "pitch": 3})


def test_mono_24k_defaults_are_sent(monkeypatch, model):
    request_input, _ = _capture_request(monkeypatch, model, {"text": "hi"})
    assert request_input["channels"] == 1
    assert request_input["sample_rate"] == 24000
    assert request_input["format"] == "wav"


# --- streaming WAV header repair --------------------------------------------


def test_repairs_placeholder_wav_header(model):
    payload = b"\x00\x01" * 100
    streamed = _wav(payload, riff_size=2147483583, data_size=2147483547)
    fixed = model._fix_wav_header(streamed)

    assert struct.unpack_from("<I", fixed, 4)[0] == len(fixed) - 8
    import io
    import wave

    with wave.open(io.BytesIO(fixed)) as handle:
        assert handle.getnframes() == len(payload) // 2
        assert handle.getframerate() == 24000


def test_leaves_wellformed_wav_untouched(model):
    good = _wav(b"\x00\x01" * 100)
    assert model._fix_wav_header(good) == good


def test_ignores_non_wav_payload(model):
    assert model._fix_wav_header(b"ID3\x04not a wav") == b"ID3\x04not a wav"


def test_saved_wav_has_correct_duration(monkeypatch, model):
    payload = b"\x00\x01" * 24000  # 1.0s at 24kHz mono 16-bit
    streamed = _wav(payload, riff_size=2147483583, data_size=2147483547)
    _, output_path = _capture_request(
        monkeypatch, model, {"text": "hi"}, audio=streamed
    )

    import wave

    with wave.open(output_path) as handle:
        assert handle.getnframes() / handle.getframerate() == pytest.approx(1.0)
