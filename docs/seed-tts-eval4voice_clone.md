# Seed-TTS Evaluation for Voice Cloning

This guide explains how to evaluate voice cloning models using the Seed-TTS evaluation dataset on UltraEval-Audio.

## Overview

The `seed_tts_eval_en` dataset is designed to evaluate voice cloning capabilities of audio generation models. This benchmark assesses how well models can synthesize speech that matches a reference voice while maintaining naturalness and clarity.

## Prerequisites

Before running the evaluation, ensure you have:

1. Completed the environment setup from the main [README.md](../README.md)
2. Activated your conda environment:
   ```bash
   conda activate aduioeval
   ```
3. Set the Python path:
   ```bash
   export PYTHONPATH=$PWD:$PYTHONPATH
   ```

## Running the Evaluation

### Basic Usage

To evaluate a voice cloning model on the Seed-TTS dataset:

```bash
python audio_evals/main.py --dataset seed_tts_eval_en --model sparkvc
```

### Available Models

You can replace `sparkvc` with any supported voice cloning model. Common options include:

- `sparkvc`: Spark Voice Cloning model
- Other TTS/voice cloning models registered in `registry/model/`


## Evaluation Results

After the evaluation completes, results will be saved in:

```
res/
└── sparkvc/                    # or your model name
    └── seed_tts_eval_en/
        ├── {timestamp}.jsonl          # Detailed results for each sample
        └── {timestamp}-overview.jsonl # Summary metrics
```

### Metrics

The evaluation typically measures:
- **Speaker Similarity**: How well the generated voice matches the reference
- **Intelligibility**: WER (Word Error Rate) if transcription is involved

## Evaluating Your Own Voice Cloning Model

To evaluate your custom voice cloning model:

1. Implement the model inference code in `audio_evals/models/`
2. Register your model in `registry/model/your_model.yaml`
3. Run the evaluation with `--model your_model`

For detailed instructions, see [how eval your model.md](how%20eval%20your%20model.md)

## Troubleshooting

### Common Issues

**Issue**: SIM model download fails

**Solution**: The WavLM model requires access to multiple sources including Google Drive, GitHub, and HuggingFace. Please ensure:
- Your internet connection is stable
- You can access these platforms without restrictions
- No firewall or proxy is blocking the downloads

**Issue**: Dataset download fails

**Solution**: For certain regions, you may need to use a HuggingFace mirror:
```bash
export HF_ENDPOINT=https://hf-mirror.com
```



## Related Resources

- [Main README](../README.md) - General usage and setup
- [How to evaluate your model](how%20eval%20your%20model.md) - Custom model integration
- [How to add a dataset](how%20add%20a%20dataset.md) - Custom dataset creation
- [How to use UTMOS & DNSMOS](how%20use%20UTMOS%2C%20DNSMOS%20eval%20speech%20quality.md) - Speech quality metrics

## Support

If you encounter any issues or have questions:
- Check the [FAQ](../FAQ.md)
- Open an issue on GitHub
- Join our Discord community: https://discord.gg/jKYuDc2M
