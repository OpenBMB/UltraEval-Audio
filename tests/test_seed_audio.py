import base64
from unittest.mock import Mock

import pytest

from audio_evals.models.TTS.seed_audio import MAX_TEXT_PROMPT_CHARS, SeedAudio


@pytest.fixture
def model():
    return SeedAudio(
        model_name="seed-audio-1.0",
        api_key="test-key",
        sample_params={"audio_config": {"format": "wav", "sample_rate": 24000}},
    )


def test_builds_plain_tts_request(model):
    request = model._build_request(
        {"text_prompt": "Hello world"},
        {"audio_config": {"format": "wav", "sample_rate": 24000}},
    )

    assert request["model"] == "seed-audio-1.0"
    assert request["text_prompt"] == "Hello world"
    assert "references" not in request


def test_accepts_bare_string_prompt(model):
    request = model._build_request("Hello world", {})

    assert request["text_prompt"] == "Hello world"


def test_builds_voice_clone_request(model, tmp_path):
    reference = tmp_path / "reference.wav"
    reference.write_bytes(b"reference audio")

    request = model._build_request(
        {
            "text_prompt": "请使用与@音频1相同的音色、口音和说话风格朗读：“你好，世界”",
            "prompt_audio": str(reference),
        },
        {},
    )

    assert request["text_prompt"].startswith("请使用与@音频1")
    assert request["references"] == [
        {"audio_data": base64.b64encode(b"reference audio").decode("ascii")}
    ]


def test_builds_speaker_request(model):
    request = model._build_request(
        {"text_prompt": "你好，世界", "speaker": "zh_male_calm"}, {}
    )

    assert request["references"] == [{"speaker": "zh_male_calm"}]


def test_empty_prompt_audio_is_ignored(model):
    request = model._build_request(
        {"text_prompt": "你好，世界", "prompt_audio": ""}, {}
    )

    assert "references" not in request


def test_forwards_request_params(model):
    request = model._build_request(
        {"text_prompt": "Hello world"},
        {"audio_config": {"format": "wav"}, "watermark": {"aigc_watermark": True}},
    )

    assert request["audio_config"] == {"format": "wav"}
    assert request["watermark"] == {"aigc_watermark": True}


def test_rejects_conflicting_voice_references(model, tmp_path):
    reference = tmp_path / "reference.wav"
    reference.write_bytes(b"reference audio")

    with pytest.raises(ValueError, match="at most one"):
        model._build_request(
            {
                "text_prompt": "Hello",
                "prompt_audio": str(reference),
                "speaker": "speaker-id",
            },
            {},
        )

    with pytest.raises(ValueError, match="at most one"):
        model._build_request(
            {
                "text_prompt": "Hello",
                "prompt_audio": str(reference),
                "references": [{"speaker": "speaker-id"}],
            },
            {},
        )

    with pytest.raises(ValueError, match="at most one"):
        model._build_request(
            {
                "text_prompt": "Hello",
                "references": [{"speaker": "speaker-id"}],
                "speaker": "speaker-id",
            },
            {},
        )


def test_rejects_dataset_columns_without_text_prompt(model):
    for payload in (
        {"text": "Hello world"},
        {"text": "Hello world", "prompt_audio": "/tmp/ref.wav"},
        {"instruction": "Speak cheerfully.", "text": "Good morning"},
        {"text": "你好", "prompt_text": "参考音频转写"},
    ):
        with pytest.raises(ValueError, match="seed_audio.yaml"):
            model._build_request(payload, {})


def test_rejects_fields_outside_request_schema(model):
    with pytest.raises(ValueError, match="language_hints"):
        model._build_request(
            {"text_prompt": "Hello world", "language_hints": ["en"]}, {}
        )


def test_requires_non_empty_text_prompt(model):
    with pytest.raises(ValueError, match="non-empty `text_prompt`"):
        model._build_request({}, {})

    with pytest.raises(ValueError, match="non-empty `text_prompt`"):
        model._build_request({"text_prompt": "   "}, {})


def test_rejects_overlong_text_prompt(model):
    with pytest.raises(ValueError, match="3000"):
        model._build_request({"text_prompt": "x" * (MAX_TEXT_PROMPT_CHARS + 1)}, {})


def test_inference_downloads_returned_audio(model):
    response = Mock()
    response.ok = True
    response.status_code = 200
    response.json.return_value = {"url": "https://example.com/audio.wav"}

    audio_response = Mock()
    audio_response.content = b"generated audio"
    audio_response.raise_for_status.return_value = None

    model.session.post = Mock(return_value=response)
    model.session.get = Mock(return_value=audio_response)

    output_path = model.inference({"text_prompt": "Hello world"})

    with open(output_path, "rb") as output:
        assert output.read() == b"generated audio"

    post_call = model.session.post.call_args
    assert post_call.kwargs["headers"]["X-Api-Key"] == "test-key"
    assert post_call.kwargs["json"]["text_prompt"] == "Hello world"
    assert post_call.kwargs["json"]["audio_config"]["format"] == "wav"


def test_inference_decodes_inline_audio(model):
    response = Mock()
    response.ok = True
    response.status_code = 200
    response.json.return_value = {
        "code": 0,
        "audio": base64.b64encode(b"generated audio").decode("ascii"),
    }
    model.session.post = Mock(return_value=response)

    output_path = model.inference({"text_prompt": "Hello world"})

    with open(output_path, "rb") as output:
        assert output.read() == b"generated audio"
