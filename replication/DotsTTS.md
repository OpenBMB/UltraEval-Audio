# DotsTTS（dots.tts-soar）复现文档与评测结果

**模型**: [dots-tts-soar](../registry/model/dots_tts.yaml) / [dots-tts-soar-fp32](../registry/model/dots_tts.yaml)
**评测日期**: 2026/08/14–2026/08/23
**模型仓库**: [dots-studio/dots.tts-soar](https://huggingface.co/dots-studio/dots.tts-soar)
**技术报告**: [dots.tts Technical Report (arXiv:2606.07080v2)](https://arxiv.org/abs/2606.07080)
**License**: Apache 2.0

**指标说明**:
- **WER/CER⬇️**: 词/字错误率，越低越好（zh 用 CER，其余 WER）
- **SIM⬆️**: 合成语音与参考音频的说话人相似度（simo: WavLM），越高越好
- **DNSMOS P808⬆️**: 语音质量 MOS，越高越好

性能格式为 `reproduced_result (official_result)`。官方结果取自技术报告 v2 Table 2（Seed-TTS-Eval）、Table 3（MiniMax 多语言）、Table 4（CV3-Eval）的 dots.tts (SOAR) 行；InstructTTSEval 官方论文未报告。

## 模型与运行时集成

- dots.tts 是 2B 参数连续潜变量自回归 TTS（Qwen2.5-1.5B 底座 + AudioVAE + AR flow-matching head + SOAR 自校正后训练），推理走**官方 PyPI 包 `dots.tts==0.3.1`**（`DotsTtsRuntime.generate`），不从源码仓库加载。
- 隔离运行时: [audio_evals/lib/DotsTTS/main.py](../audio_evals/lib/DotsTTS/main.py)（JSON-line 协议子进程）+ [requirements.txt](../audio_evals/lib/DotsTTS/requirements.txt)。
- 权重: HF `dots-studio/dots.tts-soar`（4.9 GB：`model.safetensors` + `vocoder.safetensors` + `speaker_encoder.safetensors` + tokenizer），经框架模型下载机制解析到本地后离线加载。
- 采样参数: `num_steps=10`、`guidance_scale=1.2`（SOAR 推荐设置）；`speaker_scale` 保持库默认——模型卡与 README 均未给出推荐值，实测 1.5 无效已移除。
- 精度: 主表 bf16；`dots-tts-soar-fp32` 变体按报告 §3.3.1（"All runs use float32 and have torch.compile disabled"）做 A/B。
- `torch/torchaudio` 钉在 **2.9.1**：更新版本会拉需要更新 GPU 驱动的 CUDA 13 wheel，且 torch 与 torchaudio 小版本必须严格一致，否则 torchaudio 在 import 时报 "minor versions do not match"。
- `torchaudio.load/save` 打补丁改走 `soundfile`：部分运行环境下 torchaudio 的 torchcodec/FFmpeg 后端不可用。
- `generate()` 参数走白名单（`_GENERATE_KWARGS`），模板/参数中多余键直接丢弃，不会泄漏为非法 kwarg。
- 语言标签: prompt 模板传 `lang` → runtime 给 `prompt_text` 附加 `[EN]`/`[ZH]` 等标签（`voice-clone-<lang>` 系列 prompt）。

## Seed-TTS-Eval

| 数据集 | prompt | WER/CER/%⬇️ | SIM/%⬆️ |
|---|---|---:|---:|
| seed_tts_eval_en | voice-clone-en | 1.869 (1.30) | 77.01 (77.1) |
| seed_tts_eval_zh | voice-clone-zh | 0.905 (0.94) | 80.82 (81.0) |

### A/B 实验

| 变体 | seed_tts_eval_en WER/% | SIM/% |
|---|---:|---:|
| bf16 + `voice-clone`（无语言标签） | 2.025 | 74.71 |
| bf16 + `voice-clone-en`（[EN] 标签） | 1.869 | 77.01 |
| fp32 + `voice-clone-en` | 1.837 | 76.83 |

- 语言标签对 en 有小幅改善（WER −0.16pp、SIM +2.3）；zh 侧无回归（无标签 0.916/80.83 → 有标签 0.905/80.82）。
- fp32 与 bf16 差异可忽略（WER −0.03pp），后续主表统一用 bf16。

### 官方管线交叉验证

用官方 Seed-TTS-Eval 脚本（Whisper-Large-v3 + jiwer，1088 条）对同一批 `dots-tts-soar` 合成音频复算，得 **WER 1.869%**，与 UltraEval 管线（`seed-tts-whisper`，同为 Whisper-Large-v3）的 1.8687% 一致；再对文本做一层额外处理（评测口径）后可达 **1.33% ≈ 官方 1.30%**——说明与论文的残余差距是**评测口径（文本预处理）问题**，不是 ASR 判分管线或模型质量差异。

## CV3-Eval 标准 Zero-shot

官方表仅报 zh/en/hard/hard-en 四行；其余语言为本次复现补充，无官方对照。

| 数据集 | WER/CER/%⬇️ | SIM/%⬆️ | P808 MOS⬆️ |
|---|---:|---:|---:|
| cv3_zero_shot_zh | 3.449 (3.71) | 79.74 | 3.736 |
| cv3_zero_shot_en | 5.517 (4.50) | 76.69 | 3.661 |
| cv3_zero_shot_hard_zh | 11.655 (9.22) | 77.81 | 3.638 |
| cv3_zero_shot_hard_en | 12.498 (4.49) | 75.13 | 3.679 |
| cv3_zero_shot_de | 6.449 | 79.31 | 3.665 |
| cv3_zero_shot_es | 7.811 | 78.83 | 3.703 |
| cv3_zero_shot_fr | 35.357 | 73.94 | 3.645 |
| cv3_zero_shot_it | 5.700 | 78.55 | 3.642 |
| cv3_zero_shot_ru | 14.979 | 78.40 | 3.686 |
| cv3_zero_shot_ja | 8.708 | 78.20 | 3.691 |
| cv3_zero_shot_ko | 10.498 | 79.97 | 3.741 |

- `hard_en` 与官方差距明显（12.50 vs 4.49）；`fr` 35.36 为离群值，与 MiniMax fr（19.81 vs 官方 3.56）方向一致（详见"已知差异"）。
- `cv3_emotion_zeroshot_en/zh` 未完成（仅 jsonl，无 overall 指标），对应任务失败后未补跑。

## CV3-Eval 跨语言 Zero-shot

数据集命名 `to_<lang>` 表示目标语言；官方仅报 en→zh / zh→en 两个方向，方向构成不完全可比，括号内官方值仅供参考。

| 数据集 | WER/CER/%⬇️ | SIM/%⬆️ |
|---|---:|---:|
| cv3_cross_lingual_zeroshot_to_en | 6.192 (zh→en 5.66) | 73.28 (72.8) |
| cv3_cross_lingual_zeroshot_to_zh | 16.514 (en→zh 10.75) | 74.70 (75.0) |
| cv3_cross_lingual_zeroshot_to_ja | 18.972 | 76.63 |
| cv3_cross_lingual_zeroshot_to_ko | 15.685 | 76.64 |
| cv3_cross_lingual_zeroshot_to_hard_en | 11.161 | 74.71 |
| cv3_cross_lingual_zeroshot_to_hard_zh | 19.238 | 75.22 |

## MiniMax-MLS-Test

本仓与官方均以 Whisper-Large-v3 判分（`seed-tts-whisper`），个别语言（fr/ar/ru/es）偏差较大；参考 seed-en 的口径实证（额外文本处理 1.87→1.33），疑为文本归一化口径差异，未定论。

| Language / dataset | WER/CER/%⬇️ | SIM/%⬆️ |
|---|---:|---:|
| Arabic / minimax_tts_arabic | 43.932 (36.19) | 77.14 (79.1) |
| Cantonese / minimax_tts_cantonese | 38.559 (42.32) | 85.48 (85.0) |
| Chinese / minimax_tts_chinese | 0.840 (0.77) | 82.75 (82.5) |
| Czech / minimax_tts_czech | 6.023 (4.25) | 84.00 (84.2) |
| Dutch / minimax_tts_dutch | 6.246 (1.39) | 81.96 (82.2) |
| English / minimax_tts_english | 2.979 (1.03) | 87.73 (87.5) |
| Finnish / minimax_tts_finnish | 3.760 (4.08) | 88.68 (88.3) |
| French / minimax_tts_french | 19.815 (3.56) | 77.85 (78.6) |
| German / minimax_tts_german | 0.701 (1.70) | 80.50 (80.6) |
| Greek / minimax_tts_greek | 3.701 (3.00) | 87.71 (87.6) |
| Hindi / minimax_tts_hindi | 16.594 (14.24) | 84.61 (84.7) |
| Indonesian / minimax_tts_indonesian | 2.679 (2.96) | 80.82 (80.8) |
| Italian / minimax_tts_italian | 4.020 (3.12) | 84.82 (84.7) |
| Japanese / minimax_tts_japanese | 5.633 (5.28) | 83.78 (83.7) |
| Korean / minimax_tts_korean | 4.634 (5.66) | 85.03 (83.6) |
| Polish / minimax_tts_polish | 3.126 (3.59) | 87.41 (87.8) |
| Portuguese / minimax_tts_portuguese | 2.329 (2.00) | 84.85 (84.3) |
| Romanian / minimax_tts_romanian | 3.763 (3.87) | 86.86 (87.1) |
| Russian / minimax_tts_russian | 9.118 (4.28) | 83.24 (83.2) |
| Spanish / minimax_tts_spanish | 4.276 (1.27) | 83.16 (84.0) |
| Thai / minimax_tts_thai | 8.114 (7.86) | 83.68 (83.9) |
| Turkish / minimax_tts_turkish | 5.529 (4.96) | 87.09 (87.3) |
| Ukrainian / minimax_tts_ukrainian | 1.875 (1.27) | 80.92 (81.2) |
| Vietnamese / minimax_tts_vietnamese | 4.450 (3.89) | 81.49 (81.6) |

## InstructTTSEval

官方论文未报告此基准（报告的是 EmergentTTS-Eval Overall 47.6%），下表无官方对照。判分为 Gemini LLM judge（`instruct-tts-eval-*`）。

| 数据集 | APS/% | DSD/% | RP/% | AVG/% | 失败数 |
|---|---:|---:|---:|---:|---|
| instruct-tts-eval-en | 41.84 | 51.3 | 38.6 | 43.91 | eval_fail 1（Gemini ASR 安全过滤 blockReason OTHER，fail_rate 0.033%） |
| instruct-tts-eval-zh | 43.7 | 36.9 | 39.4 | 40.0 | 0 |

## 评测配置

- **推理方式**: reference + continuation voice cloning，prompt 音频与对应文本同传 `generate(prompt_audio=..., prompt_text=..., text=...)`
- **采样参数**: `num_steps=10`、`guidance_scale=1.2`，bf16，`torch.compile` 关闭
- **Prompt**: `voice-clone-<lang>`（带语言标签，`lang` → runtime 附加 [EN]/[ZH]/… 前缀）
- **判分**: Seed-TTS-Eval / MiniMax WER/CER: Whisper-Large-v3（`seed-tts-whisper`），zh 用 Paraformer；CV3 WER/CER: CV3 官方 whisper 配置（`cv3-whisper`）；SIM: WavLM（simo）；质量: DNSMOS；InstructTTSEval: gemini-2.5-pro LLM judge

## Evaluation Commands

[1] 冒烟（每数据集先跑 5 条）:
`python audio_evals/main.py --model dots-tts-soar --dataset <dataset> --prompt voice-clone-<lang> --save res --two_phase --workers 1 --limit 5`

[2] 正式全量:
`python audio_evals/main.py --model dots-tts-soar --dataset <dataset> --prompt voice-clone-<lang> --save res --two_phase --workers 1`

[3] 失败样本补跑（`-r` 复用已成功的推理+判分，仅重跑 error 行）:
`python audio_evals/main.py --model dots-tts-soar --dataset <dataset> --prompt voice-clone-<lang> --save res --two_phase --workers 1 -r`

`<dataset>` ∈ {seed_tts_eval_en, seed_tts_eval_zh, cv3_zero_shot_*, cv3_cross_lingual_zeroshot_to_*, minimax_tts_<language>, instruct-tts-eval-en/zh}；`<lang>` 与目标语言对应（en/zh/ja/ko/de/es/fr/it/ru/ar/yue/cs/nl/fi/el/hi/id/pl/pt/ro/th/tr/uk/vi，见 [tts.yaml](../registry/prompt/tts.yaml)）。

## 已知差异与未完成项

- **seed en 1.87 vs 官方 1.30**: 官方同款 ASR 管线复算同为 1.869%，排除判分管线；对文本加一层额外处理后为 1.33（≈ 官方 1.30），确认为评测口径（文本预处理）问题，非模型质量差异。
- **法语偏差**: cv3_zero_shot_fr 35.36、minimax fr 19.81（官方 3.56），两个基准一致偏高；判分为 Whisper-Large-v3（与官方管线相同，见 `seed_tts_eval.yaml`/`minimax.yaml` evaluator），排除"ASR 管线不同"的解释。结合 seed-en 的口径实证，疑为文本归一化口径差异，也不排除模型侧法语合成问题，未定论。
- **cv3 hard_en**: 12.50 vs 官方 4.49，差距来源未定位（未见其他语言同量级偏差）。
- **cv3_emotion_zeroshot_en/zh**: 未完成（仅推理记录、无 overall 指标），后续可补跑。
- **License**: 模型以 Apache 2.0 发布，评测结果供复现与基准对比参考。
