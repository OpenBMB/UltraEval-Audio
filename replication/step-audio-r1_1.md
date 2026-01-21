# Step-Audio-R1.1 Evaluation Results

**Model**: Step-Audio-R1.1

- **Model config**: `step-audio-r1.1` in [`registry/model/gpt.yaml`](../registry/model/step.yaml)

**Evaluation Date**: 2026/01/18-2026/01/19

**Model Card**: `https://huggingface.co/stepfun-ai/Step-Audio-R1.1`

**Metrics Legend**:

- **WER (lower is better)**: Word Error Rate
- **CER (lower is better)**: Character Error Rate
- **ACC (higher is better)**: Accuracy / match rate

**Note**: The official usage example ([`examples-vllm_r1.py`](https://github.com/stepfun-ai/Step-Audio-R1/blob/main/examples-vllm_r1.py)) does not specify how to save/export the generated audio output, so we currently do not report speech-to-speech evaluation results.

---

## ASR (English)

| task    | dataset                | measure | performance | eval_cli | note                                   |
| ------- | ---------------------- | ------- | ----------- | -------- | -------------------------------------- |
| asr(en) | librispeech-test-clean | wer     | 2.24        | [1]      | Filtered abnormal samples (WER > 1000) |
| asr(en) | librispeech-dev-clean  | wer     | 1.83        | [2]      | Filtered abnormal samples (WER > 1000) |
| asr(en) | librispeech-test-other | wer     | 4.38        | [3]      | Filtered abnormal samples (WER > 1000) |
| asr(en) | librispeech-dev-other  | wer     | 3.46        | [4]      | Filtered abnormal samples (WER > 1000) |
| asr(en) | tedlium-release1       | wer     | 3.62        | [5]      | Filtered abnormal samples (WER > 1000) |

## ASR (Chinese)

| task    | dataset   | measure | performance | eval_cli | note                                   |
| ------- | --------- | ------- | ----------- | -------- | -------------------------------------- |
| asr(zh) | aishell-1 | cer     | 3.11        | [6]      | Filtered abnormal samples (WER > 1000) |
| asr(zh) | fleurs-zh | cer     | 3.49        | [7]      | Filtered abnormal samples (WER > 1000) |

## Audio Reasoning (MMAU)

> () indicate the official reported result

| task | dataset        | measure | performance | eval_cli | note |
| ---- | -------------- | ------- | ----------- | -------- | ---- |
| mmau | mmau-test-mini | acc     | 73.8(77.7)  | [8]      |      |

---

## Evaluation Commands

- [1] `python audio_evals/main.py --dataset librispeech-test-clean --model step-audio-r1.1 `
- [2] `python audio_evals/main.py --dataset librispeech-dev-clean --model step-audio-r1.1 `
- [3] `python audio_evals/main.py --dataset librispeech-test-other --model step-audio-r1.1 `
- [4] `python audio_evals/main.py --dataset librispeech-dev-other --model step-audio-r1.1 `
- [5] `python audio_evals/main.py --dataset tedlium-release1 --model step-audio-r1.1 `
- [6] `python audio_evals/main.py --dataset aishell-1 --model step-audio-r1.1 `
- [7] `python audio_evals/main.py --dataset fleurs-zh --model step-audio-r1.1 `
- [8] `python audio_evals/main.py --dataset mmau-test-mini --model step-audio-r1.1  --prompt step_audio_r1_mmau`
