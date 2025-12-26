# Qwen3-Omni Evaluation Results

**Model**: [Qwen3-Omni-30B-A3B-Instruct](../registry/model/qwen3-omni.yaml)
**Evaluation Date**: 2025/11/20
**Paper**: [Qwen3-Omni Technical Report](https://arxiv.org/abs/2509.17765)

**Metrics Legend**:
- **WER⬇️**: Word Error Rate (lower is better)
- **CER⬇️**: Character Error Rate (lower is better)
- **BLEU⬆️**: BLEU Score (higher is better)
- **ACC⬆️**: Accuracy (higher is better)

---
**Note**: Performance format: `reproduced_result(official_result)` - values in parentheses are official results from the paper.
## ASR (English)

| task | dataset | measure | performance | eval_cli | note |
|------|---------|---------|-------------|----------|------|
| asr(en) | librispeech-test-clean | wer⬇️ | 1.36(avg: 1.22) | [1] | |
| asr(en) | librispeech-dev-clean | wer⬇️ | 1.25(avg: 1.22) | [2] | |
| asr(en) | librispeech-test-other | wer⬇️ | 2.27(avg: 2.48) | [3] | |
| asr(en) | librispeech-dev-other | wer⬇️ | 2.57(avg: 2.48) | [4] | |
| asr(en) | tedlium-release1 | wer⬇️ | 2.15 | [5] | |
| asr(en) | cv-15-en | wer⬇️ | 6.00 (6.05)| [6] | |
| asr(en) | fleurs-en_us | wer⬇️ | 4.96 | [7] | |

## ASR (Chinese)

| task | dataset | measure | performance | eval_cli | note |
|------|---------|---------|-------------|----------|------|
| asr(zh) | aishell-1 | cer⬇️ |0.6 | [8] | |
| asr(zh) | cv-15-zh | cer⬇️ | 4.32(4.31)| [9] | |
| asr(zh) | fleurs-zh | cer⬇️ | 2.62(2.20) | [10] | |
| asr(zh) | WenetSpeech-test-net | cer⬇️ | 4.82(4.69) | [11] | |

## Audio Speech Translation

| task | dataset | measure | performance | eval_cli | note |
|------|---------|---------|-------------|----------|------|
| ast | covost2-zh-en | bleu⬆️ | 29.40 | [12] | |
| ast | covost2-en-zh | bleu⬆️ | 46.58| [13] | |

## Emotion Recognition

| task | dataset | measure | performance | eval_cli | note |
|------|---------|---------|-------------|----------|------|
| emo | meld-emo | acc⬆️ | 56.81| [14] | |

## Audio Generation (Speech → Speech)

| task | dataset | measure | performance | eval_cli | note |
|------|---------|---------|-------------|----------|------|
| speech-qa | Speech Web Questions | acc⬆️ | 51.50 | [15] | |
| speech-qa | Speech TriviaQA | acc⬆️ | 55.27 | [16] | |
| speech-qa | Speech CMMLU | acc⬆️ | 47.83 | [17] | |
| speech-qa | SpeechHSK | acc⬆️ | 40.27 | [18] | |
| speech-qa | Speech AlpacaEval | G-EVAL⬆️ | 67.97 | [19] | |

---

## Evaluation Commands

[1] `python audio_evals/main.py --dataset librispeech-test-clean --model qwen3-omni-audio --prompt qwen3-omni-asr-en`
[2] `python audio_evals/main.py --dataset librispeech-dev-clean --model qwen3-omni-audio --prompt qwen3-omni-asr-en`
[3] `python audio_evals/main.py --dataset librispeech-test-other --model qwen3-omni-audio --prompt qwen3-omni-asr-en`
[4] `python audio_evals/main.py --dataset librispeech-dev-other --model qwen3-omni-audio --prompt qwen3-omni-asr-en`
[5] `python audio_evals/main.py --dataset tedlium-release1 --model qwen3-omni-audio --prompt qwen3-omni-asr-en`
[6] `python audio_evals/main.py --dataset cv-15-en --model qwen3-omni-audio --prompt qwen3-omni-asr-en`
[7] `python audio_evals/main.py --dataset fleurs-en_us --model qwen3-omni-audio --prompt qwen3-omni-asr-en`
[8] `python audio_evals/main.py --dataset aishell-1 --model qwen3-omni-audio --prompt qwen3-omni-asr-zh`
[9] `python audio_evals/main.py --dataset cv-15-zh --model qwen3-omni-audio --prompt qwen3-omni-asr-zh`
[10] `python audio_evals/main.py --dataset fleurs-zh --model qwen3-omni-audio --prompt qwen3-omni-asr-zh`
[11] `python audio_evals/main.py --dataset WenetSpeech-test-net --model qwen3-omni-audio --prompt qwen3-omni-asr-zh`
[12] `python audio_evals/main.py --dataset covost2-zh-en --model qwen3-omni-audio --prompt qwen3-omni-s2tt-zh2en`
[13] `python audio_evals/main.py --dataset covost2-en-zh --model qwen3-omni-audio --prompt qwen3-omni-s2tt-en2zh`
[14] `python audio_evals/main.py --dataset meld-emo --model qwen3-omni-audio --prompt qwen3-omni-emotion`

[15] `python audio_evals/main.py --dataset speech-web-questions --model qwen3-omni-speech`

[16] `python audio_evals/main.py --dataset speech-triviaqa --model qwen3-omni-speech`
[17] `python audio_evals/main.py --dataset speech-cmmlu --model qwen3-omni-speech`
[18] `python audio_evals/main.py --dataset speech-hsk --model qwen3-omni-speech`
[19] `python audio_evals/main.py --dataset speech-alpacaeval --model qwen3-omni-speech`
