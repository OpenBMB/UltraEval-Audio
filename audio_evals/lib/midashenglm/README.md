# MidashengGLM Model

This directory contains the implementation for the MidashengGLM model, a multimodal language model that can process both text and audio inputs.

## Files

- `main.py`: Main inference script that runs the model in isolation
- `requirements.txt`: Python dependencies required for the model
- `README.md`: This documentation file

## Model Information

- **Model ID**: `mispeech/midashenglm-7b`
- **Type**: Multimodal Language Model (Text + Audio)
- **Architecture**: Based on GLM architecture with audio processing capabilities

## Usage

The model can be used through the `MidashengGLM` class in `audio_evals/models/midashenglm.py`.

### Basic Usage

```python
from audio_evals.models.midashenglm import MidashengGLM

# Initialize the model
model = MidashengGLM(
    path="mispeech/midashenglm-7b",
    sample_params={"max_length": 512, "temperature": 0.7}
)

# Text-only inference
response = model.inference("Hello, how are you?")

# Multimodal inference (text + audio)
multimodal_prompt = [
    {
        "role": "user",
        "contents": [
            {"type": "text", "value": "Caption the audio."},
            {"type": "audio", "path": "/path/to/audio.wav"}
        ]
    }
]
response = model.inference(multimodal_prompt)
```

### Input Format

The model accepts inputs in the following format:

1. **Text-only**: Simple string input
2. **Multimodal**: List of message dictionaries with role and content

Each content item can be:
- `{"type": "text", "text": "your text here"}`
- `{"type": "audio", "path": "/path/to/audio.wav"}`
- `{"type": "audio", "url": "https://example.com/audio.wav"}`
- `{"type": "audio", "audio": numpy_array}`

### Output Format

The model returns generated text responses. For multimodal inputs, it processes both text and audio to generate contextual responses.

## Dependencies

- torch >= 2.0.0
- transformers >= 4.36.0
- accelerate >= 0.20.0
- soundfile >= 0.12.0
- numpy >= 1.21.0

## Architecture

The model uses the `@isolated` decorator to run in a separate process with its own virtual environment, ensuring dependency isolation and proper resource management.

## Notes

- The model automatically downloads from HuggingFace Hub if not available locally
- Audio files are processed using soundfile library
- The model supports both local file paths and URLs for audio inputs
- CUDA is used by default for GPU acceleration
