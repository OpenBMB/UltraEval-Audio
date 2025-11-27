#!/usr/bin/env python3
"""
Example usage of CV3SpeakerSim model.

This script demonstrates how to use the CV3SpeakerSim model to compute
speaker similarity between audio pairs.
"""

import sys
import os

# Add parent directory to path if running standalone
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from audio_evals.models.cv3_speaker_sim import CV3SpeakerSim


def example_basic_usage():
    """Basic usage example with default parameters."""
    print("=== Basic Usage Example ===")

    # Initialize model with default parameters
    model = CV3SpeakerSim()

    # Prepare prompt with two audio files
    prompt = {"audios": ["/path/to/audio1.wav", "/path/to/audio2.wav"]}

    # Compute similarity
    similarity = model._inference(prompt)
    print(f"Speaker similarity: {similarity:.4f}")
    print(f"Range: [-1.0, 1.0], where 1.0 means identical speakers\n")


def example_custom_model():
    """Example using a different model variant."""
    print("=== Custom Model Example ===")

    # Use a different model variant
    model = CV3SpeakerSim(
        model_id="damo/speech_campplus_sv_zh-cn_16k-common",
        local_model_dir="/path/to/your/pretrained/models",
        device="cuda:0",
    )

    prompt = {"audios": ["/path/to/chinese_audio1.wav", "/path/to/chinese_audio2.wav"]}

    similarity = model._inference(prompt)
    print(f"Speaker similarity (Chinese model): {similarity:.4f}\n")


def example_available_models():
    """List all available model variants."""
    print("=== Available Models ===")

    models = [
        "damo/speech_campplus_sv_en_voxceleb_16k",
        "damo/speech_campplus_sv_zh-cn_16k-common",
        "damo/speech_eres2net_sv_en_voxceleb_16k",
        "damo/speech_eres2net_sv_zh-cn_16k-common",
        "damo/speech_eres2net_base_sv_zh-cn_3dspeaker_16k",  # Default
        "damo/speech_eres2net_large_sv_zh-cn_3dspeaker_16k",
    ]

    print("Supported model IDs:")
    for model_id in models:
        print(f"  - {model_id}")
    print()


def example_batch_processing():
    """Example of processing multiple audio pairs."""
    print("=== Batch Processing Example ===")

    model = CV3SpeakerSim()

    # List of audio pairs to process
    audio_pairs = [
        ("/path/to/pair1_audio1.wav", "/path/to/pair1_audio2.wav"),
        ("/path/to/pair2_audio1.wav", "/path/to/pair2_audio2.wav"),
        ("/path/to/pair3_audio1.wav", "/path/to/pair3_audio2.wav"),
    ]

    results = []
    for i, (audio1, audio2) in enumerate(audio_pairs, 1):
        prompt = {"audios": [audio1, audio2]}
        similarity = model._inference(prompt)
        results.append(similarity)
        print(f"Pair {i}: {similarity:.4f}")

    avg_similarity = sum(results) / len(results)
    print(f"\nAverage similarity: {avg_similarity:.4f}\n")


def example_integration_with_eval():
    """Example showing integration with evaluation framework."""
    print("=== Integration with Eval Framework ===")
    print(
        """
    # In your evaluation configuration YAML file:

    model:
      type: cv3_speaker_sim
      args:
        model_id: "damo/speech_eres2net_base_sv_zh-cn_3dspeaker_16k"
        local_model_dir: "pretrained"
        device: "cuda:0"

    # The model will be automatically instantiated by the framework
    # and used for speaker similarity evaluation tasks.
    """
    )


if __name__ == "__main__":
    print("CV3SpeakerSim Usage Examples\n")
    print("=" * 50)

    example_available_models()

    print("\nNote: Update the audio file paths before running the examples.\n")
    print("=" * 50)

    # Uncomment the following lines to run the examples
    # (after updating the audio file paths)

    # example_basic_usage()
    # example_custom_model()
    # example_batch_processing()
    # example_integration_with_eval()
