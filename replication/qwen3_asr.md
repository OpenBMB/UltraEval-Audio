# Qwen3-ASR-1.7B Evaluation Results

**Model**: [Qwen3-ASR-1.7B](../registry/model/qwen3-asr.yaml)
**Evaluation Date**: 2026/06/10
**Model Card**: [Qwen/Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)
**Paper**: [Qwen3-ASR Technical Report](https://arxiv.org/abs/2601.21337)

**Metrics Legend**:
- **WER⬇️**: Word Error Rate (lower is better)
- **CER⬇️**: Character Error Rate (lower is better)

---
**Note**: Performance format: `reproduced_result(official_result)` — the value in parentheses is the official number from the Qwen3-ASR model card. The framework's `res-overall.json` reports the metric as `wer(%)` for every dataset; for Chinese / dialect datasets this is computed at character level and is labeled CER below to match the official report. Official numbers are taken from the "ASR Benchmarks on Public Datasets (WER ↓)" table on the HF model card (the `Qwen3-ASR-1.7B` column).

## ASR (English)

| task | dataset | measure | performance | eval_cli | note |
|------|---------|---------|-------------|----------|------|
| asr(en) | librispeech-test-clean | wer⬇️ | 1.62(1.63) | [1] | |
| asr(en) | librispeech-test-other | wer⬇️ | 3.39(3.38) | [2] | |
| asr(en) | librispeech-dev-clean | wer⬇️ | 1.63 | [3] | card reports test split only |
| asr(en) | librispeech-dev-other | wer⬇️ | 3.06 | [4] | card reports test split only |
| asr(en) | gigaspeech | wer⬇️ | 8.29(8.45) | [5] | fail_rate 0.07%; open-asr-leaderboard: 8.74 |
| asr(en) | cv-15-en | wer⬇️ | 7.34(7.39) | [6] |  |
| asr(en) | tedlium-release1 | wer⬇️ | 2.31(4.50) | [7] | card "Tedlium" is a different split; open-asr-leaderboard tedlium: 2.28 |

## ASR (Chinese)

| task | dataset | measure | performance | eval_cli | note |
|------|---------|---------|-------------|----------|------|
| asr(zh) | WenetSpeech-test-net | cer⬇️ | 5.12(4.97) | [8] | |
| asr(zh) | WenetSpeech-test-meeting | cer⬇️ | 5.92(5.88) | [9] | |
| asr(zh) | aishell-1 | cer⬇️ | 1.52 | [10] | card reports AISHELL-2 (2.71), no AISHELL-1 |
| asr(zh) | cv-15-zh | cer⬇️ | 7.68(5.35) | [11] | 差异主因是繁简体：约10.8%的普通中文语音被转写成繁体而参考为简体，本框架打分不做繁→简归一化（`cn_tn.TextNorm` 的 `cc_mode=""`）；对转写做繁转简归一化后 CER 降至约5.56%，接近官方5.35% |
| asr(zh) | fleurs-zh | cer⬇️ | 2.73(2.41) | [12] | |

## ASR (Chinese Dialect)

| task | dataset | measure | performance | eval_cli | note |
|------|---------|---------|-------------|----------|------|
| asr(dialect) | KeSpeech | cer⬇️ | 5.06(5.10) | [13] | |

---

## Evaluation Commands

[1] `python audio_evals/main.py --dataset librispeech-test-clean --model qwen3-asr-1.7b --prompt simple-asr`
[2] `python audio_evals/main.py --dataset librispeech-test-other --model qwen3-asr-1.7b --prompt simple-asr`
[3] `python audio_evals/main.py --dataset librispeech-dev-clean --model qwen3-asr-1.7b --prompt simple-asr`
[4] `python audio_evals/main.py --dataset librispeech-dev-other --model qwen3-asr-1.7b --prompt simple-asr`
[5] `python audio_evals/main.py --dataset gigaspeech --model qwen3-asr-1.7b --prompt simple-asr`
[6] `python audio_evals/main.py --dataset cv-15-en --model qwen3-asr-1.7b --prompt simple-asr`
[7] `python audio_evals/main.py --dataset tedlium-release1 --model qwen3-asr-1.7b --prompt simple-asr`
[8] `python audio_evals/main.py --dataset WenetSpeech-test-net --model qwen3-asr-1.7b --prompt simple-asr`
[9] `python audio_evals/main.py --dataset WenetSpeech-test-meeting --model qwen3-asr-1.7b --prompt simple-asr`
[10] `python audio_evals/main.py --dataset aishell-1 --model qwen3-asr-1.7b --prompt simple-asr`
[11] `python audio_evals/main.py --dataset cv-15-zh --model qwen3-asr-1.7b --prompt simple-asr`
[12] `python audio_evals/main.py --dataset fleurs-zh --model qwen3-asr-1.7b --prompt simple-asr`
[13] `python audio_evals/main.py --dataset KeSpeech --model qwen3-asr-1.7b --prompt simple-asr`
