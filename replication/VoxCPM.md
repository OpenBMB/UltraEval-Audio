# VoxCPM Evaluation Results

**Model**: [VoxCPM-0.5B](../registry/model/voxcpm.yaml)
**Evaluation Date**: 2025/12/08
**Paper/Repo**: [openbmb/VoxCPM-0.5B](https://huggingface.co/openbmb/VoxCPM-0.5B)

**Metrics Legend**:
- **WER⬇️**: Word Error Rate (lower is better)
- **CER⬇️**: Character Error Rate (lower is better)
- **SIM⬆️**: Speaker Similarity (higher is better)
- **P808_MOS⬆️**: DNSMOS P.808 Mean Opinion Score (higher is better)

---

## Seed-TTS-Eval Benchmark

| task | dataset | WER/CER⬇️ | SIM⬆️ | eval_cli | note |
|------|---------|-----------|-------|----------|------|
| tts | seed_tts_eval_en | 1.81(1.85) | 73.23(72.9) | [1] | |
| tts | seed_tts_eval_zh | 1.04(0.93) | 77.10(77.2) | [2] | |

---

## CV3 Benchmark (Zero-Shot)

| task | dataset | WER/CER⬇️ | SIM⬆️ | P808_MOS⬆️ | eval_cli | note |
|------|---------|-----------|-------|------------|----------|------|
| tts | cv3_zero_shot_en | 6.64(4.04) | 67.93 | 3.78 | [3] | |
| tts | cv3_zero_shot_zh | 3.40(3.40) | 71.84 | 3.85 | [4] | |
| tts | cv3_zero_shot_hard_en | 7.45(7.89) | 64.99(64.3) | 3.93(3.74) | [5] | |
| tts | cv3_zero_shot_hard_zh | 9.95(12.9) | 66.78(66.1) | 3.74(3.59) | [6] | |

---

## Evaluation Commands

[1] `python audio_evals/main.py --dataset seed_tts_eval_en --model voxcpm-vc`

[2] `python audio_evals/main.py --dataset seed_tts_eval_zh --model voxcpm-vc`

[3] `python audio_evals/main.py --dataset cv3_zero_shot_en --model voxcpm-vc`

[4] `python audio_evals/main.py --dataset cv3_zero_shot_zh --model voxcpm-vc`

[5] `python audio_evals/main.py --dataset cv3_zero_shot_hard_en --model voxcpm-vc`

[6] `python audio_evals/main.py --dataset cv3_zero_shot_hard_zh --model voxcpm-vc`
