# CosyVoice2 Evaluation Results

**Model**: [CosyVoice2-0.5B](../registry/model/cosyvoice.yaml)
**Evaluation Date**: 2025/12/08
**Paper/Repo**: [FunAudioLLM/CosyVoice2-0.5B](https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B)

**Metrics Legend**:
- **WER⬇️**: Word Error Rate (lower is better)
- **CER⬇️**: Character Error Rate (lower is better)
- **SIM⬆️**: Speaker Similarity (higher is better)
- **P808_MOS⬆️**: DNSMOS P.808 Mean Opinion Score (higher is better)

---

**Note**:
- Performance format: `reproduced_result(official_result)` - values in parentheses are official results from the paper.
- Replicating CosyVoice2 results is challenging as the model weights and inference code have been updated over time.
## Seed-TTS-Eval Benchmark

| task | dataset | WER/CER⬇️ | SIM⬆️ | eval_cli | note |
|------|---------|-----------|-------|----------|------|
| tts | seed_tts_eval_en | 3.24(2.57) | 65.36(65.2) | [1] | |
| tts | seed_tts_eval_zh | 1.28(1.45) | 75.03(75.3) | [2] | |

---

## CV3 Benchmark (Zero-Shot)

| task | dataset | WER/CER⬇️ | SIM⬆️ | P808_MOS⬆️ | eval_cli | note |
|------|---------|-----------|-------|------------|----------|------|
| tts | cv3_zero_shot_en | 7.60(6.32) | 71.13 | 3.81 | [3] | |
| tts | cv3_zero_shot_zh | 4.02(4.08) | 77.90 | 3.85 | [4] | |
| tts | cv3_zero_shot_hard_en | 13.91(11.96) | 70.93(66.7) | 3.93(3.95) | [5] | |
| tts | cv3_zero_shot_hard_zh | 8.35(12.58) | 76.33(72.6) | 3.81(3.81) | [6] | https://github.com/FunAudioLLM/CV3-Eval/issues/4|

---

# Long-TTS-Eval Benchmark

> Update (2025/12/19): results may change as [FunAudioLLM/CosyVoice](https://github.com/FunAudioLLM/CosyVoice.git) is updated.
>
> the official result from Long-TTS-Eval

| task | dataset | WER(%)⬇️ | eval_cli | note |
|------|---------|----------|----------|------|
| tts | long_tts_eval_en | 7.17(14.80) | [7] | model: cosyvoice2-official |
| tts | long_tts_eval_zh | 7.11(5.27) | [8] | model: cosyvoice2-official |
| tts | long_tts_eval_hard_en | 37.61(43.48) | [9] | model: cosyvoice2-official |
| tts | long_tts_eval_hard_zh | 34.31(32.76) | [10] | model: cosyvoice2-official |


## Evaluation Commands

[1] `python audio_evals/main.py --dataset seed_tts_eval_en --model cosyvoice2-vc`

[2] `python audio_evals/main.py --dataset seed_tts_eval_zh --model cosyvoice2-vc`

[3] `python audio_evals/main.py --dataset cv3_zero_shot_en --model cosyvoice2-vc`

[4] `python audio_evals/main.py --dataset cv3_zero_shot_zh --model cosyvoice2-vc`

[5] `python audio_evals/main.py --dataset cv3_zero_shot_hard_en --model cosyvoice2-vc`

[6] `python audio_evals/main.py --dataset cv3_zero_shot_hard_zh --model cosyvoice2-vc`

[7] `python audio_evals/main.py --dataset long_tts_eval_en --model cosyvoice2-official`

[8] `python audio_evals/main.py --dataset long_tts_eval_zh --model cosyvoice2-official`

[9] `python audio_evals/main.py --dataset long_tts_eval_hard_en --model cosyvoice2-official`

[10] `python audio_evals/main.py --dataset long_tts_eval_hard_zh --model cosyvoice2-official`
