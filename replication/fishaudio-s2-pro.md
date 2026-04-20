# Fish Speech S2 Pro 评测结果 / Evaluation Results

**模型 / Model**: [fishaudio-s2-pro](../registry/model/fishaudio_s2.yaml)
**评测日期 / Evaluation Date**: 2026/04
**Paper/Repo**: [fishaudio/s2-pro](https://huggingface.co/fishaudio/s2-pro)

**指标说明 / Metrics**:
- **WER⬇️**: Word Error Rate — 词错误率，越低越好 / lower is better
- **CER⬇️**: Character Error Rate — 字符错误率，越低越好 / lower is better
- **SIM⬆️**: Speaker Similarity — 说话人相似度，越高越好 / higher is better

---

## Seed-TTS-Eval Benchmark

| task | dataset | WER/CER⬇️ | SIM⬆️ | eval_cli | note |
|------|---------|-----------|-------|----------|------|
| tts | seed_tts_eval_en | 1.82 | 64.94 | [1] | |
| tts | seed_tts_eval_zh | 1.03 | 72.84 | [2] | |

> **说明**: 上述 Seed-TTS-Eval 结果（EN WER=1.82, ZH CER=1.03）与官方 README / 技术报告中公布的数值（EN=0.99, ZH=0.54）存在较大差距。原因是论文中公布的数值使用的是 fishaudio **线上服务**的模型，而非开源模型；本评测基于公开权重 `fishaudio/s2-pro` 及 `fish-speech` 仓库公开的本地推理路径完成，因此无法与论文数值对齐。相关讨论参考 [fishaudio/fish-speech#1268](https://github.com/fishaudio/fish-speech/issues/1268)。
>
> **Note (EN)**: The Seed-TTS-Eval numbers above (EN WER=1.82, ZH CER=1.03) differ noticeably from the values reported in the official README / technical report (EN=0.99, ZH=0.54). The reason is that the paper's reported numbers were produced by fishaudio's **online service** model, not the open-source release. This evaluation is based on the publicly released weights `fishaudio/s2-pro` and the local inference pipeline provided in the `fish-speech` repository, so the results are not directly comparable to the paper. See the related discussion in [fishaudio/fish-speech#1268](https://github.com/fishaudio/fish-speech/issues/1268).

---

## MiniMax TTS 多语言 Benchmark / MiniMax TTS Multilingual Benchmark

> 来源数据集 / Source dataset: [MiniMaxAI/TTS-Multilingual-Test-Set](https://huggingface.co/datasets/MiniMaxAI/TTS-Multilingual-Test-Set)

| task | language | dataset | WER/CER⬇️ | SIM⬆️ | eval_cli |
|------|----------|---------|-----------|-------|----------|
| tts | Arabic | minimax_tts_arabic | 8.02 (WER) | 73.26 | [3] |
| tts | Cantonese | minimax_tts_cantonese | 46.15 (CER) | 74.75 | [4] |
| tts | Chinese | minimax_tts_chinese | 1.08 (CER) | 76.90 | [5] |
| tts | Czech | minimax_tts_czech | 6.57 (WER) | 76.68 | [6] |
| tts | Dutch | minimax_tts_dutch | 1.40 (WER) | 71.59 | [7] |
| tts | English | minimax_tts_english | 2.39 (WER) | 78.71 | [8] |
| tts | Finnish | minimax_tts_finnish | 7.69 (WER) | 80.43 | [9] |
| tts | French | minimax_tts_french | 4.18 (WER) | 68.12 | [10] |
| tts | German | minimax_tts_german | 1.04 (WER) | 71.04 | [11] |
| tts | Greek | minimax_tts_greek | 12.45 (WER) | 79.71 | [12] |
| tts | Hindi | minimax_tts_hindi | 22.38 (WER) | 78.84 | [13] |
| tts | Indonesian | minimax_tts_indonesian | 2.82 (WER) | 73.29 | [14] |
| tts | Japanese | minimax_tts_japanese | 3.65 (CER) | 75.99 | [15] |
| tts | Korean | minimax_tts_korean | 1.59 (CER) | 75.32 | [16] |
| tts | Polish | minimax_tts_polish | 2.41 (WER) | 79.02 | [17] |
| tts | Portuguese | minimax_tts_portuguese | 1.53 (WER) | 79.31 | [18] |
| tts | Romanian | minimax_tts_romanian | 15.83 (WER) | 75.55 | [19] |
| tts | Russian | minimax_tts_russian | 3.89 (WER) | 76.31 | [20] |
| tts | Spanish | minimax_tts_spanish | 1.56 (WER) | 73.80 | [21] |
| tts | Thai | minimax_tts_thai | 7.59 (CER) | 74.87 | [22] |
| tts | Turkish | minimax_tts_turkish | 2.03 (WER) | 77.99 | [23] |
| tts | Vietnamese | minimax_tts_vietnamese | 15.74 (WER) | 70.87 | [24] |

---

## Evaluation Commands

[1] `python audio_evals/main.py --dataset seed_tts_eval_en --model fishaudio-s2-pro`

[2] `python audio_evals/main.py --dataset seed_tts_eval_zh --model fishaudio-s2-pro`

[3] `python audio_evals/main.py --dataset minimax_tts_arabic --model fishaudio-s2-pro`

[4] `python audio_evals/main.py --dataset minimax_tts_cantonese --model fishaudio-s2-pro`

[5] `python audio_evals/main.py --dataset minimax_tts_chinese --model fishaudio-s2-pro`

[6] `python audio_evals/main.py --dataset minimax_tts_czech --model fishaudio-s2-pro`

[7] `python audio_evals/main.py --dataset minimax_tts_dutch --model fishaudio-s2-pro`

[8] `python audio_evals/main.py --dataset minimax_tts_english --model fishaudio-s2-pro`

[9] `python audio_evals/main.py --dataset minimax_tts_finnish --model fishaudio-s2-pro`

[10] `python audio_evals/main.py --dataset minimax_tts_french --model fishaudio-s2-pro`

[11] `python audio_evals/main.py --dataset minimax_tts_german --model fishaudio-s2-pro`

[12] `python audio_evals/main.py --dataset minimax_tts_greek --model fishaudio-s2-pro`

[13] `python audio_evals/main.py --dataset minimax_tts_hindi --model fishaudio-s2-pro`

[14] `python audio_evals/main.py --dataset minimax_tts_indonesian --model fishaudio-s2-pro`

[15] `python audio_evals/main.py --dataset minimax_tts_japanese --model fishaudio-s2-pro`

[16] `python audio_evals/main.py --dataset minimax_tts_korean --model fishaudio-s2-pro`

[17] `python audio_evals/main.py --dataset minimax_tts_polish --model fishaudio-s2-pro`

[18] `python audio_evals/main.py --dataset minimax_tts_portuguese --model fishaudio-s2-pro`

[19] `python audio_evals/main.py --dataset minimax_tts_romanian --model fishaudio-s2-pro`

[20] `python audio_evals/main.py --dataset minimax_tts_russian --model fishaudio-s2-pro`

[21] `python audio_evals/main.py --dataset minimax_tts_spanish --model fishaudio-s2-pro`

[22] `python audio_evals/main.py --dataset minimax_tts_thai --model fishaudio-s2-pro`

[23] `python audio_evals/main.py --dataset minimax_tts_turkish --model fishaudio-s2-pro`

[24] `python audio_evals/main.py --dataset minimax_tts_vietnamese --model fishaudio-s2-pro`
