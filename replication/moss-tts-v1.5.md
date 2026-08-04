# MOSS-TTS-v1.5 评测结果 / Evaluation Results

**模型 / Model**: [moss-tts-v1.5](../registry/model/moss_tts.yaml) ([OpenMOSS-Team/MOSS-TTS-v1.5](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-v1.5), MossTTSDelay-8B)
**评测日期 / Evaluation Date**: 2026/07/21–2026/08/03
**Paper/Repo**: [OpenMOSS/MOSS-TTS](https://github.com/OpenMOSS/MOSS-TTS)

**指标说明 / Metrics**:
- **WER⬇️**: Word Error Rate — 词错误率，越低越好 / lower is better
- **CER⬇️**: Character Error Rate — 字符错误率，越低越好 / lower is better
- **SIM⬆️**: Speaker Similarity — 说话人相似度，越高越好 / higher is better
- **P808_MOS⬆️**: DNSMOS P.808 Mean Opinion Score，越高越好 / higher is better

---

**说明 / Note**:
- Seed-TTS-Eval 采用 `复现值(官方值)` 的格式；官方值来自 [MOSS-TTS model card](https://github.com/OpenMOSS/MOSS-TTS/blob/main/docs/moss_tts_model_card.md) 中的 MossTTSDelay-8B 结果。
- CV3 与 MiniMax 表格为无锡集群（speech-wx / H100）复现值；结果取对应数据集最新且完整的 `overall.json`。
- `overall.json` 中的 `P808_MOS(%)` 按 100 倍存储，表格已换算为正常 MOS 范围。

## Seed-TTS-Eval Benchmark

| task | dataset | WER/CER⬇️ | SIM⬆️ | eval_cli | note |
|------|---------|-----------|-------|----------|------|
| tts | seed_tts_eval_en | 2.69(1.84) | 67.95(70.86) | [1] | 1088 samples; fail_rate: 0% |
| tts | seed_tts_eval_zh | 1.49(1.37) | 76.13(76.98) | [2] | 2020 samples; fail_rate: 0% |

> **复现对齐情况 / Reproduction note**: 中文结果与官方值基本对齐（CER +0.12，SIM -0.85）；英文结果存在更明显差距（WER +0.85，SIM -2.91）。现有结果文件不足以确认差距来自推理参数、语言标签或评测环境，本文不对根因作推断。

---

## CV3 Benchmark (Zero-Shot)

| task | dataset | WER/CER⬇️ | SIM⬆️ | P808_MOS⬆️ | eval_cli | note |
|------|---------|-----------|-------|------------|----------|------|
| tts | cv3_zero_shot_en | 5.50 (WER) | 66.70 | 3.75 | [3] | fail_rate: 0% |
| tts | cv3_zero_shot_zh | 3.93 (CER) | 72.95 | 3.80 | [4] | fail_rate: 0% |
| tts | cv3_zero_shot_hard_en | 6.39 (WER) | 66.36 | 3.85 | [5] | fail_rate: 0% |
| tts | cv3_zero_shot_hard_zh | 8.39 (CER) | 70.27 | 3.78 | [6] | fail_rate: 0% |
| tts | cv3_zero_shot_de | 7.85 (WER) | 69.99 | 3.71 | [7] | fail_rate: 0% |
| tts | cv3_zero_shot_es | 4.20 (WER) | 71.63 | 3.69 | [8] | fail_rate: 0% |
| tts | cv3_zero_shot_fr | 12.43 (WER) | 67.58 | 3.68 | [9] | fail_rate: 0% |
| tts | cv3_zero_shot_it | 5.83 (WER) | 69.40 | 3.68 | [10] | fail_rate: 0% |
| tts | cv3_zero_shot_ja | 6.44 (CER) | 68.90 | 3.72 | [11] | fail_rate: 0% |
| tts | cv3_zero_shot_ko | 6.71 (CER) | 72.66 | 3.80 | [12] | fail_rate: 0% |
| tts | cv3_zero_shot_ru | 6.86 (WER) | 68.93 | 3.73 | [13] | fail_rate: 0% |

---

## MiniMax TTS 多语言 Benchmark / MiniMax TTS Multilingual Benchmark

> 来源数据集 / Source dataset: [MiniMaxAI/TTS-Multilingual-Test-Set](https://huggingface.co/datasets/MiniMaxAI/TTS-Multilingual-Test-Set)

| task | language | dataset | WER/CER⬇️ | SIM⬆️ | eval_cli | note |
|------|----------|---------|-----------|-------|----------|------|
| tts | Arabic | minimax_tts_arabic | 10.25 (WER) | 74.12 | [14] | fail_rate: 0% |
| tts | Cantonese | minimax_tts_cantonese | 35.25 (CER) | 79.68 | [15] | fail_rate: 0% |
| tts | Chinese | minimax_tts_chinese | 1.23 (CER) | 81.03 | [16] | fail_rate: 0% |
| tts | Czech | minimax_tts_czech | 5.42 (WER) | 76.98 | [17] | fail_rate: 0% |
| tts | Dutch | minimax_tts_dutch | 1.22 (WER) | 73.16 | [18] | fail_rate: 0% |
| tts | English | minimax_tts_english | 2.20 (WER) | 80.55 | [19] | fail_rate: 0% |
| tts | Finnish | minimax_tts_finnish | 8.60 (WER) | 85.85 | [20] | fail_rate: 0% |
| tts | French | minimax_tts_french | 5.43 (WER) | 65.87 | [21] | fail_rate: 0% |
| tts | German | minimax_tts_german | 0.55 (WER) | 75.94 | [22] | fail_rate: 0% |
| tts | Greek | minimax_tts_greek | 4.50 (WER) | 82.00 | [23] | fail_rate: 0% |
| tts | Hindi | minimax_tts_hindi | 20.86 (WER) | 83.63 | [24] | fail_rate: 0% |
| tts | Italian | minimax_tts_italian | 1.73 (WER) | 77.13 | [25] | fail_rate: 0% |
| tts | Japanese | minimax_tts_japanese | 3.82 (CER) | 78.78 | [26] | fail_rate: 0% |
| tts | Korean | minimax_tts_korean | 2.20 (CER) | 80.46 | [27] | fail_rate: 0% |
| tts | Polish | minimax_tts_polish | 2.67 (WER) | 83.02 | [28] | fail_rate: 0% |
| tts | Portuguese | minimax_tts_portuguese | 2.33 (WER) | 81.99 | [29] | fail_rate: 0% |
| tts | Romanian | minimax_tts_romanian | 4.05 (WER) | 78.20 | [30] | fail_rate: 0% |
| tts | Russian | minimax_tts_russian | 5.24 (WER) | 76.72 | [31] | fail_rate: 0% |
| tts | Spanish | minimax_tts_spanish | 1.30 (WER) | 77.04 | [32] | fail_rate: 0% |
| tts | Thai | minimax_tts_thai | 2.82 (CER) | 80.72 | [33] | fail_rate: 0% |
| tts | Turkish | minimax_tts_turkish | 2.07 (WER) | 80.62 | [34] | fail_rate: 0% |
| tts | Vietnamese | minimax_tts_vietnamese | 2.88 (WER) | 75.85 | [35] | fail_rate: 0% |

---

## Evaluation Commands

[1] `python audio_evals/main.py --dataset seed_tts_eval_en --model moss-tts-v1.5`

[2] `python audio_evals/main.py --dataset seed_tts_eval_zh --model moss-tts-v1.5`

[3] `python audio_evals/main.py --dataset cv3_zero_shot_en --model moss-tts-v1.5`

[4] `python audio_evals/main.py --dataset cv3_zero_shot_zh --model moss-tts-v1.5`

[5] `python audio_evals/main.py --dataset cv3_zero_shot_hard_en --model moss-tts-v1.5`

[6] `python audio_evals/main.py --dataset cv3_zero_shot_hard_zh --model moss-tts-v1.5`

[7] `python audio_evals/main.py --dataset cv3_zero_shot_de --model moss-tts-v1.5`

[8] `python audio_evals/main.py --dataset cv3_zero_shot_es --model moss-tts-v1.5`

[9] `python audio_evals/main.py --dataset cv3_zero_shot_fr --model moss-tts-v1.5`

[10] `python audio_evals/main.py --dataset cv3_zero_shot_it --model moss-tts-v1.5`

[11] `python audio_evals/main.py --dataset cv3_zero_shot_ja --model moss-tts-v1.5`

[12] `python audio_evals/main.py --dataset cv3_zero_shot_ko --model moss-tts-v1.5`

[13] `python audio_evals/main.py --dataset cv3_zero_shot_ru --model moss-tts-v1.5`

[14] `python audio_evals/main.py --dataset minimax_tts_arabic --model moss-tts-v1.5`

[15] `python audio_evals/main.py --dataset minimax_tts_cantonese --model moss-tts-v1.5`

[16] `python audio_evals/main.py --dataset minimax_tts_chinese --model moss-tts-v1.5`

[17] `python audio_evals/main.py --dataset minimax_tts_czech --model moss-tts-v1.5`

[18] `python audio_evals/main.py --dataset minimax_tts_dutch --model moss-tts-v1.5`

[19] `python audio_evals/main.py --dataset minimax_tts_english --model moss-tts-v1.5`

[20] `python audio_evals/main.py --dataset minimax_tts_finnish --model moss-tts-v1.5`

[21] `python audio_evals/main.py --dataset minimax_tts_french --model moss-tts-v1.5`

[22] `python audio_evals/main.py --dataset minimax_tts_german --model moss-tts-v1.5`

[23] `python audio_evals/main.py --dataset minimax_tts_greek --model moss-tts-v1.5`

[24] `python audio_evals/main.py --dataset minimax_tts_hindi --model moss-tts-v1.5`

[25] `python audio_evals/main.py --dataset minimax_tts_italian --model moss-tts-v1.5`

[26] `python audio_evals/main.py --dataset minimax_tts_japanese --model moss-tts-v1.5`

[27] `python audio_evals/main.py --dataset minimax_tts_korean --model moss-tts-v1.5`

[28] `python audio_evals/main.py --dataset minimax_tts_polish --model moss-tts-v1.5`

[29] `python audio_evals/main.py --dataset minimax_tts_portuguese --model moss-tts-v1.5`

[30] `python audio_evals/main.py --dataset minimax_tts_romanian --model moss-tts-v1.5`

[31] `python audio_evals/main.py --dataset minimax_tts_russian --model moss-tts-v1.5`

[32] `python audio_evals/main.py --dataset minimax_tts_spanish --model moss-tts-v1.5`

[33] `python audio_evals/main.py --dataset minimax_tts_thai --model moss-tts-v1.5`

[34] `python audio_evals/main.py --dataset minimax_tts_turkish --model moss-tts-v1.5`

[35] `python audio_evals/main.py --dataset minimax_tts_vietnamese --model moss-tts-v1.5`
