# Qwen3-TTS 复现文档与评测结果

**模型**: [Qwen3-TTS](../registry/model/qwen3tts.yaml)  
**评测日期**: 2026/02  

**指标说明**:
- **WER⬇️ / CER⬇️**: ASR 识别错误率（越低越好）
- **SIM⬆️**: 说话人相似度（越高越好）
- **DNSMOS⬆️**: 语音质量打分（越高越好，范围 0–5）

---

## Seed-TTS-Eval（Voice Clone）复现结果

**Note**: 性能格式为 `reproduced_result(official_result)`，括号内为论文/官方结果（如有）。

> 下面命令会自动下载权重到 `init_model/` 并运行 voice clone；首次运行会较慢。

| 模型 | SEED-test-en (WER⬇️) | SEED-test-en (SIM⬆️) | SEED-test-zh (CER⬇️) | SEED-test-zh (SIM⬆️) | eval_cli |
|---|---:|---:|---:|---:|---|
| Qwen3-TTS-12Hz-1.7B-Base-official-infer-params | 1.58 (1.24) | 71.24 | 0.87 (0.78) | 76.89 | en:[1] zh:[2] |
| Qwen3-TTS-12Hz-1.7B-Base-official-infer-params-xvec_only | 1.56 (1.24) | 59.61 | 0.78 (0.78) | 72.92 | en:[3] zh:[4] |
| Qwen3-TTS-12Hz-0.6B-Base-official-infer-params | 1.69 (1.32) | 70.55 | 1.01 (0.92) | 76.48 | en:[5] zh:[6] |

## CV3-Eval（Zero-shot Voice Clone）复现结果

> CV3-Eval 在本项目中按 split 分开跑（`cv3_zero_shot_{en,zh}` 与 `cv3_zero_shot_hard_{en,zh}`），表格为汇总展示。

| 模型 | zh CER/%⬇️ | en WER/%⬇️ | hard-zh CER/%⬇️ | hard-zh SIM/%⬆️ | hard-zh DNSMOS⬆️ | hard-en WER/%⬇️ | hard-en SIM/%⬆️ | hard-en DNSMOS⬆️ | eval_cli |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Qwen3-TTS-12Hz-1.7B-Base-official-infer-params | 3.12±0.07 | 3.77±0.19 | 11.33±1.43 | 70.13 | 3.83 | 7.90±1.77 | 66.06 | 3.91 | zh:[8] en:[7] hard-zh:[10] hard-en:[9] |
| Qwen3-TTS-12Hz-0.6B-Base-official-infer-params | 3.40±0.09 | 33.91±13.06 | 10.70±1.06 | 69.72 | 3.82 | 10.70±2.90 | 67.04 | 3.88 | zh:[12] en:[11] hard-zh:[14] hard-en:[13] |

---

## Evaluation Commands

[1] `python audio_evals/main.py --dataset seed_tts_eval_en --model qwen3-tts-1.7b-base --prompt qwen3-tts-voice-clone-english --use_model_pool --workers 8`  
[2] `python audio_evals/main.py --dataset seed_tts_eval_zh --model qwen3-tts-1.7b-base --prompt qwen3-tts-voice-clone-chinese --use_model_pool --workers 8`  
[3] `python audio_evals/main.py --dataset seed_tts_eval_en --model qwen3-tts-12hz-1.7b-base-xvec_only --prompt qwen3-tts-voice-clone-english --use_model_pool --workers 8`  
[4] `python audio_evals/main.py --dataset seed_tts_eval_zh --model qwen3-tts-12hz-1.7b-base-xvec_only --prompt qwen3-tts-voice-clone-chinese --use_model_pool --workers 8`  
[5] `python audio_evals/main.py --dataset seed_tts_eval_en --model qwen3-tts-0.6b-base --prompt qwen3-tts-voice-clone-english --use_model_pool --workers 8`  
[6] `python audio_evals/main.py --dataset seed_tts_eval_zh --model qwen3-tts-0.6b-base --prompt qwen3-tts-voice-clone-chinese --use_model_pool --workers 8`  

[7] `python audio_evals/main.py --dataset cv3_zero_shot_en --model qwen3-tts-1.7b-base --prompt qwen3-tts-voice-clone-english --use_model_pool --workers 8`  
[8] `python audio_evals/main.py --dataset cv3_zero_shot_zh --model qwen3-tts-1.7b-base --prompt qwen3-tts-voice-clone-chinese --use_model_pool --workers 8`  
[9] `python audio_evals/main.py --dataset cv3_zero_shot_hard_en --model qwen3-tts-1.7b-base --prompt qwen3-tts-voice-clone-english --use_model_pool --workers 8`  
[10] `python audio_evals/main.py --dataset cv3_zero_shot_hard_zh --model qwen3-tts-1.7b-base --prompt qwen3-tts-voice-clone-chinese --use_model_pool --workers 8`  

[11] `python audio_evals/main.py --dataset cv3_zero_shot_en --model qwen3-tts-0.6b-base --prompt qwen3-tts-voice-clone-english --use_model_pool --workers 8`  
[12] `python audio_evals/main.py --dataset cv3_zero_shot_zh --model qwen3-tts-0.6b-base --prompt qwen3-tts-voice-clone-chinese --use_model_pool --workers 8`  
[13] `python audio_evals/main.py --dataset cv3_zero_shot_hard_en --model qwen3-tts-0.6b-base --prompt qwen3-tts-voice-clone-english --use_model_pool --workers 8`  
[14] `python audio_evals/main.py --dataset cv3_zero_shot_hard_zh --model qwen3-tts-0.6b-base --prompt qwen3-tts-voice-clone-chinese --use_model_pool --workers 8`


