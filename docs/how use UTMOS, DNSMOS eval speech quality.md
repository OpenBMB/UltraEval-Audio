# Comprehensive Guide to Speech Quality Evaluation Using UTMOS and DNSMOS

## Introduction
UTMOS and DNSMOS (P.835, P.808) are state-of-the-art metrics for evaluating speech quality in audio processing applications. This comprehensive guide provides detailed instructions on effectively utilizing these evaluation tools to assess speech generation quality.

## Prerequisites
Before proceeding, ensure your Python environment is properly configured:
```bash
export PYTHONPATH=$PWD:$PYTHONPATH
```

## Evaluating Speech Generation Models

### Running the Evaluation
To evaluate speech generation models using our benchmark suite, simply specify the speech quality task:

```bash
CUDA_VISIBLE_DEVICES=0 python audio_evals/main.py --dataset llama-questions --model MiniCPMo2_6-speech --task speech-quality
```

### Understanding the Results
The evaluation generates a detailed JSON output containing multiple quality metrics for each sample:

```json
{
    "type": "eval",
    "id": 1,
    "data": {
        "pred": "/tmp/tmp_r333js9.wav",
        "ref": "/audio_web_questions/audio/test_1.wav",
        "OVRL": 3.5930655279805093,
        "SIG": 3.7941088203624935,
        "BAK": 4.246229203642303,
        "P808_MOS": 4.20117712020874,
        "P_SIG": 4.740789147662223,
        "P_BAK": 4.817738828193093,
        "P_OVRL": 4.650278837922578,
        "utmos": 4.461420178413391
    }
}
```

### Metric Interpretations

#### UTMOS Score
- The `utmos` field represents the UTMOS score for the predicted speech file
- Score range: 1-5 (higher indicates better quality)

#### DNSMOS Scores
The DNSMOS evaluation provides multiple quality dimensions:
- `OVRL`: DNSMOS P.835  (1-5)
- `P808_MOS`: DNSMOS P.808 score

For more detailed information about DNSMOS metrics, please refer to the [official documentation](https://github.com/microsoft/DNS-Challenge/tree/master/DNSMOS).

### Important Notes

#### Supported Models
The following models currently support speech generation capabilities:
- MiniCPMo2_6-speech
- qwen2.5-omni-speech
- kimiaudio-speech
- gpt4o_speech
- gpt4o_speech_ms
- glm-4-voice
- step-speech

#### Available Benchmarks
We support the following speech evaluation benchmarks:
- speech-web-questions
- llama-questions
- speech-chatbot-alpaca-eval
- speech-triviaqa

## Direct Usage of UTMOS

To use UTMOS directly in your Python code:

```python
from audio_evals.registry import registry

# Initialize the UTMOS model
model = registry.get_mode("utmos-en")

# Evaluate an audio file
audio_path = "path/to/your/audio.wav"
quality_score = model.inference(audio_path)
print(f"UTMOS Score: {quality_score}")
```

## Direct Usage of DNSMOS

To use DNSMOS directly in your Python code:

```python
from audio_evals.registry import registry

# Initialize the DNSMOS model
model = registry.get_mode("dnsmos")

# Evaluate an audio file
audio_path = "path/to/your/audio.wav"
quality_metrics = model.inference({"audio": audio_path})
print("DNSMOS Metrics:", quality_metrics)
```
