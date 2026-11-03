"""Evaluators for EmergentTTS-Eval.

The benchmark (https://arxiv.org/abs/2505.23009) uses a Large Audio Language
Model as a pairwise judge: the generated audio is compared against the
``gpt-4o-mini-tts`` (alloy) baseline shipped with the dataset, and the judge
returns a 0/1/2 winner plus 0-3 scores for both systems.

This module provides :class:`EmergentTTSEvalGemini`, the pairwise win-rate
judge. It is intentionally tolerant of judge failures: the evaluator returns
``None`` instead of raising, so the aggregation can report how many samples
were actually judged (matching the official
``judger_parsing_failed_count``).
"""

import json
import logging
import os
import re
import tempfile
import zlib
from typing import Dict, Optional

import numpy as np

from audio_evals.evaluator.base import Evaluator
from audio_evals.evaluator.emergent_tts_eval_prompts import get_win_rate_prompts

logger = logging.getLogger(__name__)


def _iter_json_objects(text: str):
    """Yield balanced ``{...}`` substrings, skipping braces inside strings."""
    depth = 0
    start = None
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                yield text[start : index + 1]
                start = None


def extract_judge_json(text: Optional[str]) -> Optional[dict]:
    """Extract the judge's JSON object from a possibly noisy response."""
    if not text:
        return None

    candidates = []
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fenced:
        candidates.append(fenced.group(1))
    candidates.append(text.strip())
    candidates.extend(_iter_json_objects(text))

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue
        if isinstance(parsed, dict) and "winner" in parsed:
            return parsed
    return None


def _as_int(value):
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _validate_judgement(result: dict) -> dict:
    """Normalize and validate the fields used for scoring."""
    winner = _as_int(result.get("winner"))
    if winner not in (0, 1, 2):
        raise ValueError("judge response has no valid `winner`: {!r}".format(result))

    normalized = dict(result)
    normalized["winner"] = winner
    for key in ("score_1", "score_2"):
        normalized[key] = _as_int(result.get(key))
    return normalized


def _sample_seed(sample_id) -> int:
    """Reproduce the official A/B randomization seed (the source row index)."""
    if sample_id is None:
        return 0
    if isinstance(sample_id, (int, np.integer)) and not isinstance(sample_id, bool):
        return int(sample_id)
    # Fall back to a stable hash when the dataset does not provide a row index.
    return zlib.crc32(str(sample_id).encode("utf-8"))


class EmergentTTSEvalGemini(Evaluator):
    """
    Pairwise model-as-a-judge evaluator for EmergentTTS-Eval.

    Compares the generated audio (``pred``) with the dataset's baseline audio
    (``WavPath``) using a Gemini judge and the official judging prompts.
    System order is randomized with the sample ``id`` as seed, exactly like the
    official implementation, to neutralize position bias.

    The request mirrors the official ``score_existing_results.py``: both audios
    are resampled to 16 kHz mono PCM-16 WAV, the judge runs at temperature 0
    with ``topP=0.9`` and ``maxOutputTokens=32768``, no ``thinkingConfig`` and
    no ``safetySettings``.  Deviating from any of these changes the score
    calibration and makes results incomparable with the official file.

    Args:
        model_name       : Registry name of the judge model.  Use the pinned
                           ``gemini-3.8-flash`` entry so the gateway serves
                           the same deployment as the official run.
        thinking_budget  : Gemini thinking budget.  ``None`` (default) omits
                           ``thinkingConfig``, matching the official scorer.
        max_output_tokens: Output token budget for the judge (official: 32768).
        max_retries      : Attempts before giving up on one sample.
        randomize_order  : Randomize which system is presented first.
    """

    def __init__(
        self,
        model_name: str = "gemini-3.8-flash",
        thinking_budget: Optional[int] = None,
        max_output_tokens: int = 32768,
        max_retries: int = 5,
        randomize_order: bool = True,
    ):
        from audio_evals.registry import registry

        self.model_name = model_name
        self.model = registry.get_model(model_name)
        self.thinking_budget = thinking_budget
        self.max_output_tokens = max_output_tokens
        self.max_retries = max_retries
        self.randomize_order = randomize_order

    def _predicted_speech_index(self, sample_id) -> int:
        """1 when the model is system 1, 2 when it is system 2."""
        if not self.randomize_order:
            return 1
        return 2 if np.random.default_rng(_sample_seed(sample_id)).random() > 0.5 else 1

    @staticmethod
    def _to_wav_16k(audio_path: str) -> str:
        """Resample to 16 kHz mono PCM-16 WAV, exactly like the official scorer."""
        import librosa
        import soundfile as sf

        waveform, _ = librosa.load(audio_path, sr=16000, mono=True)
        waveform = np.clip(waveform, -1.0, 1.0)
        fd, wav_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            sf.write(
                wav_path,
                (waveform * 32767).astype(np.int16),
                16000,
                format="WAV",
                subtype="PCM_16",
            )
        except Exception:
            os.unlink(wav_path)
            raise
        return wav_path

    def _infer(self, prompt_struct):
        kwargs = {
            "temperature": 0,
            "topP": 0.9,
            "maxOutputTokens": self.max_output_tokens,
        }
        if self.thinking_budget is not None:
            kwargs["thinkingConfig"] = {
                "thinkingBudget": self.thinking_budget,
                "includeThoughts": True,
            }
        return self.model.inference(prompt_struct, **kwargs)

    def _eval(self, pred: str, label: str, **kwargs) -> Dict[str, any]:
        model_audio = str(pred)
        baseline_audio = str(kwargs.get("WavPath") or label)
        text = kwargs.get("text") or label
        category = kwargs.get("category", "")
        sample_id = kwargs.get("id")

        if not os.path.isfile(model_audio):
            raise FileNotFoundError(
                "Generated audio file not found: {}".format(model_audio)
            )
        if not os.path.isfile(baseline_audio):
            raise FileNotFoundError(
                "Baseline audio file not found: {}".format(baseline_audio)
            )

        system_prompt, user_message, post_audio_1_message = get_win_rate_prompts(
            text, category
        )

        predicted_speech_index = self._predicted_speech_index(sample_id)
        wav_paths = []
        try:
            # The official scorer always sends 16 kHz mono PCM-16 WAV, even
            # when the generated file is an mp3 at a different sample rate.
            model_wav = self._to_wav_16k(model_audio)
            wav_paths.append(model_wav)
            baseline_wav = self._to_wav_16k(baseline_audio)
            wav_paths.append(baseline_wav)
            if predicted_speech_index == 1:
                audio_1, audio_2 = model_wav, baseline_wav
            else:
                audio_1, audio_2 = baseline_wav, model_wav

            prompt_struct = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "contents": [
                        {"type": "text", "value": user_message},
                        {"type": "audio", "value": audio_1},
                        {"type": "text", "value": post_audio_1_message},
                        {"type": "audio", "value": audio_2},
                    ],
                },
            ]

            last_response = None
            last_error = None
            for attempt in range(self.max_retries):
                try:
                    last_response = self._infer(prompt_struct)
                    result = extract_judge_json(last_response)
                    if result is None:
                        raise ValueError("could not parse a JSON object with `winner`")
                    result = _validate_judgement(result)

                    winner = result["winner"]
                    win = (
                        1.0
                        if winner == predicted_speech_index
                        else (0.5 if winner == 0 else 0.0)
                    )
                    model_score = (
                        result["score_1"]
                        if predicted_speech_index == 1
                        else result["score_2"]
                    )
                    baseline_score = (
                        result["score_2"]
                        if predicted_speech_index == 1
                        else result["score_1"]
                    )
                    return {
                        "win": win,
                        "tie": 1 if winner == 0 else 0,
                        "winner": winner,
                        "predicted_speech_index": predicted_speech_index,
                        "model_score": model_score,
                        "baseline_score": baseline_score,
                        "score_1": result["score_1"],
                        "score_2": result["score_2"],
                        "category": category,
                        "judge_model": self.model_name,
                        "judge_comparison": result.get("system_comparison", ""),
                        "judge_raw_output": last_response,
                    }
                except Exception as exc:  # noqa: BLE001 - retried, surfaced below
                    last_error = exc
                    logger.warning(
                        "[EmergentTTS-Eval] judge attempt %d/%d failed for id=%r: %s",
                        attempt + 1,
                        self.max_retries,
                        sample_id,
                        exc,
                    )

            logger.error(
                "[EmergentTTS-Eval] judge failed after %d attempts for id=%r: %s",
                self.max_retries,
                sample_id,
                last_error,
            )
        finally:
            for wav_path in wav_paths:
                try:
                    os.unlink(wav_path)
                except OSError:
                    pass

        return {
            "win": None,
            "tie": None,
            "winner": -1,
            "predicted_speech_index": predicted_speech_index,
            "category": category,
            "judge_model": self.model_name,
            "judge_raw_output": last_response,
        }
