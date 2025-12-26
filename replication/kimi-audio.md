# Kimi-Audio Evaluation Results

**Model**: [Kimi-Audio-7B-Instruct](../registry/model/moonshot.yaml)
**Evaluation Date**: 2025/11/20
**Paper**: [Kimi-Audio Technical Report](https://arxiv.org/abs/2504.18425)

**Metrics Legend**:
- **WER⬇️**: Word Error Rate (lower is better)
- **CER⬇️**: Character Error Rate (lower is better)
- **BLEU⬆️**: BLEU Score (higher is better)
- **ACC⬆️**: Accuracy (higher is better)

---

## ASR (English)

| task | dataset | measure | performance | eval_cli | note |
|------|---------|---------|-------------|----------|------|
| asr(en) | librispeech-test-clean | wer⬇️ | 1.28 | [1] | |
| asr(en) | librispeech-dev-clean | wer⬇️ | 1.18 | [2] | |
| asr(en) | librispeech-test-other | wer⬇️ | 2.44 | [3] | |
| asr(en) | librispeech-dev-other | wer⬇️ | 2.35 | [4] | |
| asr(en) | tedlium-release1 | wer⬇️ | 2.96 | [5] | |
| asr(en) | cv-15-en | wer⬇️ | 7.09 | [6] | |
| asr(en) | fleurs-en_us | wer⬇️ | 5.06 | [7] | |

## ASR (Chinese)

| task | dataset | measure | performance | eval_cli | note |
|------|---------|---------|-------------|----------|------|
| asr(zh) | aishell-1 | cer⬇️ | 0.60 | [8] | |
| asr(zh) | cv-15-zh | cer⬇️ | 5.73 | [9] | |
| asr(zh) | fleurs-zh | cer⬇️ | 3.08 | [10] | |
| asr(zh) | WenetSpeech-test-net | cer⬇️ | 5.56 | [11] | |

## Audio Speech Translation

| task | dataset | measure | performance | eval_cli | note |
|------|---------|---------|-------------|----------|------|
| ast | covost2-zh-en | bleu⬆️ | 18.30| [12] | |
| ast | covost2-en-zh | bleu⬆️ | 36.61| [13] | |

## Audio Generation (Speech → Speech)

| task | dataset | measure | performance | eval_cli | note |
|------|---------|---------|-------------|----------|------|
| speech-qa | Speech Web Questions | acc⬆️ | 33.69 | [14] | |
| speech-qa | Speech TriviaQA | acc⬆️ | 38.20 | [15] | |
| speech-qa | Speech CMMLU | acc⬆️ | 71.25 | [16] | |
| speech-qa | SpeechHSK | acc⬆️ | 97.42 | [17] | |
| speech-qa | Speech AlpacaEval | G-EVAL⬆️ | 34.40 | [18] | |


## Evaluation Commands

[1] `python audio_evals/main.py --dataset librispeech-test-clean --model kimiaudio --prompt kimi-audio-asr-en`
[2] `python audio_evals/main.py --dataset librispeech-dev-clean --model kimiaudio --prompt kimi-audio-asr-en`
[3] `python audio_evals/main.py --dataset librispeech-test-other --model kimiaudio --prompt kimi-audio-asr-en`
[4] `python audio_evals/main.py --dataset librispeech-dev-other --model kimiaudio --prompt kimi-audio-asr-en`
[5] `python audio_evals/main.py --dataset tedlium-release1 --model kimiaudio --prompt kimi-audio-asr-en`
[6] `python audio_evals/main.py --dataset cv-15-en --model kimiaudio --prompt kimi-audio-asr-en`
[7] `python audio_evals/main.py --dataset fleurs-en_us --model kimiaudio --prompt kimi-audio-asr-en`
[8] `python audio_evals/main.py --dataset aishell-1 --model kimiaudio --prompt kimi-audio-asr-zh`
[9] `python audio_evals/main.py --dataset cv-15-zh --model kimiaudio --prompt kimi-audio-asr-zh`
[10] `python audio_evals/main.py --dataset fleurs-zh --model kimiaudio --prompt kimi-audio-asr-zh`
[11] `python audio_evals/main.py --dataset WenetSpeech-test-net --model kimiaudio --prompt kimi-audio-asr-zh`
[12] `python audio_evals/main.py --dataset covost2-zh-en --model kimiaudio`
[13] `python audio_evals/main.py --dataset covost2-en-zh --model kimiaudio`

[14] `python audio_evals/main.py --dataset speech-web-questions --model kimiaudio-speech`

[15] `python audio_evals/main.py --dataset speech-triviaqa --model kimiaudio-speech`
[16] `python audio_evals/main.py --dataset speech-cmmlu --model kimiaudio-speech`
[17] `python audio_evals/main.py --dataset speech-hsk --model kimiaudio-speech`
[18] `python audio_evals/main.py --dataset speech-alpacaeval --model kimiaudio-speech`
