# CV3 Speaker Similarity Model

This module integrates the 3D-Speaker project's speaker verification models into the UltraEval-Audio framework.

## Overview

The CV3SpeakerSim model computes speaker similarity by extracting embeddings from audio files and calculating their cosine similarity. It supports multiple pre-trained models from the 3D-Speaker project.

## Architecture

The integration follows the framework's standard pattern:

- **Model Wrapper** (`audio_evals/models/cv3_speaker_sim.py`):
  - Inherits from `OfflineModel`
  - Uses `@isolated` decorator for process isolation
  - Handles communication via stdin/stdout with UUID-based request tracking

- **Inference Script** (`audio_evals/lib/cv3_speaker_sim/speaker_sim.py`):
  - Interactive mode supporting concurrent requests
  - Loads 3D-Speaker models (ERes2Net, CAMPPlus, etc.)
  - Processes audio pairs and returns similarity scores

## Supported Models

| Model ID | Language | Architecture | Embedding Size |
|----------|----------|--------------|----------------|
| `damo/speech_campplus_sv_en_voxceleb_16k` | English | CAMPPlus | 512 |
| `damo/speech_campplus_sv_zh-cn_16k-common` | Chinese | CAMPPlus | 192 |
| `damo/speech_eres2net_sv_en_voxceleb_16k` | English | ERes2Net | 192 |
| `damo/speech_eres2net_sv_zh-cn_16k-common` | Chinese | ERes2Net | 192 |
| `damo/speech_eres2net_base_sv_zh-cn_3dspeaker_16k` | Chinese | ERes2Net Base | 512 |
| `damo/speech_eres2net_large_sv_zh-cn_3dspeaker_16k` | Chinese | ERes2Net Large | 512 |

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Install 3D-Speaker library:
```bash
# Clone and install from the 3D-Speaker repository
git clone https://github.com/alibaba-damo-academy/3D-Speaker.git
cd 3D-Speaker/speakerlab
pip install -e .
```

3. Download pretrained models:
```bash
# Models should be placed in the following structure:
# pretrained/
#   ├── speech_campplus_sv_en_voxceleb_16k/
#   │   └── campplus_voxceleb.bin
#   ├── speech_eres2net_base_sv_zh-cn_3dspeaker_16k/
#   │   └── eres2net_base_model.ckpt
#   └── ...
```

You can download models from [ModelScope](https://www.modelscope.cn/models).

## Usage

### Basic Usage

```python
from audio_evals.models.cv3_speaker_sim import CV3SpeakerSim

# Initialize model
model = CV3SpeakerSim(
    model_id="damo/speech_eres2net_base_sv_zh-cn_3dspeaker_16k",
    local_model_dir="pretrained",
    device="cuda:0"
)

# Compute similarity between two audio files
prompt = {
    'audios': ['/path/to/audio1.wav', '/path/to/audio2.wav']
}

similarity = model._inference(prompt)
print(f"Speaker similarity: {similarity:.4f}")  # Range: [-1.0, 1.0]
```

### Configuration in YAML

```yaml
model:
  type: cv3_speaker_sim
  args:
    model_id: "damo/speech_eres2net_base_sv_zh-cn_3dspeaker_16k"
    local_model_dir: "pretrained"
    device: "cuda:0"
```

### Direct Script Usage

You can also run the inference script directly:

```bash
python speaker_sim.py \
  --model_id damo/speech_eres2net_base_sv_zh-cn_3dspeaker_16k \
  --local_model_dir pretrained \
  --device cuda:0
```

Then send requests in the format:
```
<uuid>->/path/to/audio1.wav,/path/to/audio2.wav
```

## Output

The model returns a float value in the range `[-1.0, 1.0]`:
- `1.0`: Identical speakers (perfect match)
- `0.0`: No correlation
- `-1.0`: Opposite (very rare in practice)

Typically, scores > 0.5 indicate the same speaker, while scores < 0.3 indicate different speakers. The exact threshold depends on the model and application.

## Performance

- **Processing Time**: ~50-100ms per audio pair (depending on audio length and model size)
- **Memory Usage**: ~2-4GB GPU memory (depending on model variant)
- **Supported Audio**: 16kHz sample rate, mono or stereo (will use first channel)

## Comparison with WavLM

Both CV3SpeakerSim and WavLM compute speaker similarity, but differ in:

| Feature | CV3SpeakerSim | WavLM |
|---------|---------------|-------|
| Architecture | ERes2Net / CAMPPlus | ECAPA-TDNN + WavLM |
| Model Source | 3D-Speaker | Custom |
| Model Variants | 6 variants | 1 variant |
| Language Support | EN / ZH | Universal |
| Typical Score Range | 0.2-0.9 | 0.3-0.95 |

## Troubleshooting

### Model Not Found Error
Ensure the model files are downloaded and placed in the correct directory structure under `local_model_dir`.

### CUDA Out of Memory
Try using a smaller model variant or reduce batch size if processing multiple pairs.

### Import Error: speakerlab
Make sure you've installed the 3D-Speaker library:
```bash
pip install git+https://github.com/alibaba-damo-academy/3D-Speaker.git#subdirectory=speakerlab
```

## References

- [3D-Speaker Project](https://github.com/alibaba-damo-academy/3D-Speaker)
- [ERes2Net Paper](https://arxiv.org/abs/2305.12838)
- [CAMPPlus Paper](https://arxiv.org/abs/2303.00332)

## License

This integration follows the Apache License 2.0, consistent with the 3D-Speaker project.
