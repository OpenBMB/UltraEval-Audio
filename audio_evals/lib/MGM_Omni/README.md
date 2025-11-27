# MGM-Omni Integration

This directory contains the integration for MGM-Omni TTS model in UltraEval-Audio.

## Model Information

MGM-Omni is a text-to-speech model with voice cloning capabilities. It supports:
- Voice cloning from reference audio
- Multi-language speech synthesis
- High-quality audio generation at 24kHz

## Installation

The dependencies will be automatically installed when you first run the model. You can also manually install them:

```bash
pip install -r audio_evals/lib/MGM_Omni/requirements.txt
```

**Note**: You also need to install the MGM model package from the official repository:
```bash
# Clone and install MGM repository
git clone https://github.com/your-mgm-repo/MGM.git
cd MGM
pip install -e .
```

## Usage

### Basic TTS (Full precision)
```bash
CUDA_VISIBLE_DEVICES=0 python audio_evals/main.py --dataset your-dataset --model mgm-omni-tts
```

### TTS with 8-bit quantization
```bash
CUDA_VISIBLE_DEVICES=0 python audio_evals/main.py --dataset your-dataset --model mgm-omni-tts-8bit
```

### TTS with 4-bit quantization
```bash
CUDA_VISIBLE_DEVICES=0 python audio_evals/main.py --dataset your-dataset --model mgm-omni-tts-4bit
```

## Configuration

You can customize the model by editing `registry/model/mgm_omni.yaml`:

- `path`: Path or HuggingFace model ID for MGM-Omni model
- `ref_audio`: Path to reference audio file for voice cloning
- `ref_audio_text`: (Optional) Transcript of reference audio. If not provided, Whisper will be used for ASR
- `cosyvoice_path`: (Optional) Path to CosyVoice model if needed
- `temperature`: Sampling temperature (default: 0.3)
- `max_new_tokens`: Maximum tokens to generate (default: 8192)
- `load_8bit`: Enable 8-bit quantization
- `load_4bit`: Enable 4-bit quantization

## Model Architecture

MGM-Omni uses:
- Qwen2VL conversation template
- Speech token encoding/decoding
- Reference audio conditioning for voice cloning
- Audio generation at 24kHz sample rate

## Reference

Original inference code adapted from MGM-Omni repository.
