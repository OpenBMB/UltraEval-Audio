# VoxCPM2 复现文档与评测结果

**模型**: [VoxCPM2](../registry/model/voxcpm2.yaml)
**评测日期**: 2026/07/26–2026/07/27
**模型仓库**: [openbmb/VoxCPM2](https://huggingface.co/openbmb/VoxCPM2)
**技术报告**: [VoxCPM2 Technical Report](https://arxiv.org/abs/2606.06928)

**指标说明**:
- **WER/CER⬇️**: 词/字错误率，越低越好
- **SIM⬆️**: 合成语音与参考音频的说话人相似度，越高越好
- **P808 MOS⬆️**: DNSMOS P.808 语音质量，越高越好

性能格式为 `reproduced_result (official_result)`。官方结果取自技术报告 Table 3、5–7 及[官方仓库](https://github.com/OpenBMB/VoxCPM#performance)。

## Seed-TTS-Eval

| 数据集 | WER/CER/%⬇️ | SIM/%⬆️ |
|---|---:|---:|
| seed_tts_eval_en | 1.643 (1.84) | 75.253 (75.3) |
| seed_tts_eval_zh | 0.920 (0.97) | 79.403 (79.5) |


## CV3-Eval 标准 Zero-shot

官方 CV3 表仅报告 WER/CER；SIM 和 P808 MOS 为本次复现的补充指标。

| 数据集 | WER/CER/%⬇️ | SIM/%⬆️ | P808 MOS⬆️ |
|---|---:|---:|---:|
| cv3_zero_shot_zh | 3.505 (3.65) | 74.839 | 3.740 |
| cv3_zero_shot_en | 5.278 (5.00) | 71.598 | 3.657 |
| cv3_zero_shot_hard_zh | 8.327 (8.55) | 72.218 | 3.695 |
| cv3_zero_shot_hard_en | 7.222 (8.48) | 69.055 | 3.789 |
| cv3_zero_shot_ja | 5.947 (5.96) | 73.027 | 3.656 |
| cv3_zero_shot_ko | 5.29 (5.69) | 74.266 | 3.728 |
| cv3_zero_shot_de | 4.629 (4.77) | 74.461 | 3.617 |
| cv3_zero_shot_es | 3.879 (3.80) | 74.711 | 3.635 |
| cv3_zero_shot_fr | 9.727 (9.85) | 70.702 | 3.581 |
| cv3_zero_shot_it | 3.848 (4.25) | 73.545 | 3.595 |
| cv3_zero_shot_ru | 5.037 (5.21) | 72.581 | 3.652 |


## MiniMax-MLS-Test

| Language / dataset | WER/CER/%⬇️ | SIM/%⬆️ |
|---|---:|---:|
| Arabic / minimax_tts_arabic | 12.797 (13.046) | 79.129 (79.1) |
| Cantonese / minimax_tts_cantonese | 38.334 (38.584) | 83.610 (83.5) |
| Chinese / minimax_tts_chinese | 1.040 (1.136) | 82.581 (82.5) |
| Czech / minimax_tts_czech | 24.567 (24.132) | 77.777 (78.3) |
| Dutch / minimax_tts_dutch | 0.740 (0.913) | 80.395 (80.8) |
| English / minimax_tts_english | 2.371 (2.289) | 85.265 (85.4) |
| Finnish / minimax_tts_finnish | 2.458 (2.632) | 87.744 (89.0) |
| French / minimax_tts_french | 4.515 (4.534) | 73.678 (73.5) |
| German / minimax_tts_german | 1.054 (0.679) | 79.874 (80.3) |
| Greek / minimax_tts_greek | 2.660 (2.844) | 86.010 (86.0) |
| Hindi / minimax_tts_hindi | 17.968 (19.699) | 86.168 (85.6) |
| Indonesian / minimax_tts_indonesian | 1.302 (1.084) | 78.866 (80.0) |
| Italian / minimax_tts_italian | 1.987 (1.563) | 77.941 (78.0) |
| Japanese / minimax_tts_japanese | 4.201 (4.628) | 82.567 (82.8) |
| Korean / minimax_tts_korean | 3.651 (1.962) | 83.577 (83.3) |
| Polish / minimax_tts_polish | 1.379 (1.141) | 87.777 (88.4) |
| Portuguese / minimax_tts_portuguese | 2.036 (1.938) | 83.645 (83.7) |
| Romanian / minimax_tts_romanian | 20.772 (21.577) | 80.494 (79.7) |
| Russian / minimax_tts_russian | 3.035 (3.634) | 80.713 (81.1) |
| Spanish / minimax_tts_spanish | 1.482 (1.438) | 82.648 (83.1) |
| Thai / minimax_tts_thai | 2.215 (2.961) | 83.862 (84.0) |
| Turkish / minimax_tts_turkish | 0.611 (0.817) | 86.987 (87.1) |
| Ukrainian / minimax_tts_ukrainian | 6.942 (6.316) | 78.570 (79.8) |
| Vietnamese / minimax_tts_vietnamese | 1.963 (3.307) | 78.732 (80.6) |


## 评测配置

- 推理方式: 参考音频与对应文本同时用于 reference + continuation voice cloning
- 推理参数: `cfg_value=2.0`，`inference_timesteps=10`；主表均未启用去噪
- MiniMax WER/CER: Whisper Large V3（中文例外使用 Paraformer），SIM: WavLM Large
- CV3 WER/CER: CV3 官方评测配置，SIM: ERes2Net，质量: DNSMOS

## Evaluation Commands

[1] Seed-TTS-Eval：
`python audio_evals/main.py --model voxcpm2 --dataset <seed_tts_eval_en|seed_tts_eval_zh> --save res --two_phase --workers 1`

[2] CV3 标准/跨语言/情感任务：
`python audio_evals/main.py --model voxcpm2 --dataset <cv3_dataset> --prompt voice-clone --save res --two_phase --workers 1`

[3] MiniMax-MLS-Test 任务：
`python audio_evals/main.py --model voxcpm2 --dataset <minimax_tts_language> --prompt voice-clone --save res --two_phase --workers 1`
