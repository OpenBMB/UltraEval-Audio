# MGM-Omni Evaluation Results

**Model**: [MGM-Omni](../registry/model/mgm_omni.yaml)
**Evaluation Date**: 2025/11/27
**Paper**: [MGM-Omni Technical Report](https://arxiv.org/abs/2410.07949)

**Metrics Legend**:
- **WER⬇️**: Word Error Rate (lower is better)
- **SIM⬆️**: Speaker Similarity (higher is better)

---

## Speech Generation (seed-tts-eval)

**Note**: Performance format: `reproduced_result(official_result)` - values in parentheses are official results from the paper.

| task | dataset | measure | performance | eval_cli | note |
|------|---------|---------|-------------|----------|------|
| voice-clone | seedtts-vc-zh | cer⬇️ |1.27(1.18) | [1] | Chinese Voice Cloning |
| voice-clone | seedtts-vc-zh | simo⬇️ | 75.53(75.8) | [1] | Chinese Voice Cloning |
| voice-clone | seedtts-vc-en | wer⬇️ | 2.53(2.22)| [2] | English Voice Cloning |
| voice-clone | seedtts-vc-en | wer⬇️ | 68.5(68.6)| [2] | English Voice Cloning |

---

## Evaluation Commands

[1] `python audio_evals/main.py --dataset seed_tts_eval_zh --model mgm-omni`
[2] `python audio_evals/main.py --dataset seed_tts_eval_en --model mgm-omni`
