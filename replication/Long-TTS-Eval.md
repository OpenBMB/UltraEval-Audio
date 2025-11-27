# Long-TTS-Eval Benchmark Results

**Benchmark**: [Long-TTS-Eval Dataset](../registry/dataset/long-tts-eval.yaml) | [Task Config](../registry/eval_task/tts.yaml)
**Evaluation Date**: 2025/11/27
**Paper**: [Long-TTS-Eval: A Benchmark for Evaluating Long-form Text-to-Speech](https://arxiv.org/abs/2505.23009)

**Metrics Legend**:
- **WER⬇️**: Word Error Rate (lower is better)
- **CER⬇️**: Character Error Rate (lower is better)

---

## Long-TTS-Eval Standard Set

**Note**: Performance format: `reproduced_result(official_result)` - values in parentheses are official results from the paper.

| task | sub | measure | performance | eval_cli | note |
|------|---------|---------|-------------|----------|------|
| tts | long_tts_eval_zh | cer⬇️ | 7.23(5.58) | [1] |  |
| tts | long_tts_eval_en | wer⬇️ | 4.69(4.98) | [2] | |
| tts | long_tts_eval_hard_zh | cer⬇️ | 24.33(23.58) | [3] | https://github.com/dvlab-research/MGM-Omni/issues/6 |
| tts | long_tts_eval_hard_en | wer⬇️ | 32.84(26.26) | [4] | |

---

## Evaluation Commands

[1] `python audio_evals/main.py --dataset long_tts_eval_zh --model mgm-omni-tts-zh`
[2] `python audio_evals/main.py --dataset long_tts_eval_en --model mgm-omni-tts`
[3] `python audio_evals/main.py --dataset long_tts_eval_hard_zh --model mgm-omni-tts-zh`
[4] `python audio_evals/main.py --dataset long_tts_eval_hard_en --model mgm-omni-tts`
