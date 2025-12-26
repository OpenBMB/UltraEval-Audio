# GLM-TTS Evaluation Results

**Model**: `glmtts` ([config](../registry/model/glmtts.yaml))
**Evaluation Date**: 2025/12 (from `res/glmtts/*/res-overall.json`)
**Paper/Repo**: [zai-org/GLM-TTS](https://github.com/zai-org/GLM-TTS)

**Metrics Legend**:
- **WER⬇️**: Word Error Rate (lower is better)
- **CER⬇️**: Character Error Rate (lower is better)
- **SIM⬆️**: Speaker Similarity (higher is better)
- **P808_MOS⬆️**: DNSMOS P.808 Mean Opinion Score (higher is better)

---


## Seed-TTS-Eval Benchmark

| task | dataset | WER/CER⬇️ | SIM⬆️ | eval_cli | note |
|------|---------|-----------|-------|----------|------|
| tts | seed_tts_eval_en | 2.48 | 67.25 | [1] | |
| tts | seed_tts_eval_zh | 1.06(1.03) | 75.98(76.1) | [2] | |

---

## CV3 Benchmark (Zero-Shot)

| task | dataset | WER/CER⬇️ | SIM⬆️ | P808_MOS⬆️ | eval_cli | note |
|------|---------|-----------|-------|------------|----------|------|
| tts | cv3_zero_shot_en | 6.61 | 72.08 | 3.70 | [3] | |
| tts | cv3_zero_shot_zh | 3.61 | 77.98 | 3.73 | [4] | |
| tts | cv3_zero_shot_hard_en | 9.40 | 73.43 | 3.73 | [5] | fail_rate: 3.12% |
| tts | cv3_zero_shot_hard_zh | 9.14 | 77.79 | 3.64 | [6] | |

---

## Evaluation Commands

[1] `python audio_evals/main.py --dataset seed_tts_eval_en --model glmtts`
[2] `python audio_evals/main.py --dataset seed_tts_eval_zh --model glmtts`

[3] `python audio_evals/main.py --dataset cv3_zero_shot_en --model glmtts`
[4] `python audio_evals/main.py --dataset cv3_zero_shot_zh --model glmtts`
[5] `python audio_evals/main.py --dataset cv3_zero_shot_hard_en --model glmtts`
[6] `python audio_evals/main.py --dataset cv3_zero_shot_hard_zh --model glmtts`
