# IndexTTS Evaluation Results

**Model**: [IndexTTS2](../registry/model/indextts.yaml)
**Evaluation Date**: 2025/12/08
**Paper/Repo**: [IndexTeam/IndexTTS-2](https://huggingface.co/IndexTeam/IndexTTS-2)

**Metrics Legend**:
- **WER⬇️**: Word Error Rate (lower is better)
- **CER⬇️**: Character Error Rate (lower is better)
- **SIM⬆️**: Speaker Similarity (higher is better)

---

## Seed-TTS-Eval Benchmark

### IndexTTS2

| task | dataset | WER/CER⬇️ | SIM⬆️ | eval_cli | note |
|------|---------|-----------|-------|----------|------|
| tts | seed_tts_eval_en | 2.08(1.52) | 70.40 | [3] | |
| tts | seed_tts_eval_zh | 1.04(1.01)| 76.04 | [4] | |

---

## CV3 Benchmark (Zero-Shot)

### IndexTTS2

| task | dataset | WER/CER⬇️ | SIM⬆️ | eval_cli | note |
|------|---------|-----------|-------|----------|------|
| tts | cv3_zero_shot_en | 4.32 | 73.60 | [9] | |
| tts | cv3_zero_shot_zh | 3.63 | 78.00 | [10] | |
| tts | cv3_zero_shot_hard_en | 8.59 | 74.01 | [11] | |
| tts | cv3_zero_shot_hard_zh | 8.71 | 77.21 | [12] | |

---

## Evaluation Commands

### IndexTTS2

[3] `python audio_evals/main.py --dataset seed_tts_eval_en --model indextts2`

[4] `python audio_evals/main.py --dataset seed_tts_eval_zh --model indextts2`

[9] `python audio_evals/main.py --dataset cv3_zero_shot_en --model indextts2`

[10] `python audio_evals/main.py --dataset cv3_zero_shot_zh --model indextts2`

[11] `python audio_evals/main.py --dataset cv3_zero_shot_hard_en --model indextts2`

[12] `python audio_evals/main.py --dataset cv3_zero_shot_hard_zh --model indextts2`
