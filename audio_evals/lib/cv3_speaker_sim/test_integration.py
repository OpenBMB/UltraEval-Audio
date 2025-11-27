#!/usr/bin/env python3
"""
Integration test for CV3SpeakerSim model.

This script performs basic sanity checks to ensure the model integration works correctly.
"""

import sys
import os
import tempfile
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))


def create_dummy_audio(duration=1.0, sample_rate=16000, frequency=440):
    """Create a dummy audio file for testing purposes."""
    try:
        import torchaudio
        import torch

        # Generate a sine wave
        t = torch.linspace(0, duration, int(sample_rate * duration))
        audio = torch.sin(2 * np.pi * frequency * t).unsqueeze(0)

        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        torchaudio.save(temp_file.name, audio, sample_rate)
        temp_file.close()

        return temp_file.name
    except Exception as e:
        print(f"Error creating dummy audio: {e}")
        return None


def test_model_initialization():
    """Test model initialization with default parameters."""
    print("Test 1: Model Initialization")
    print("-" * 50)

    try:
        from audio_evals.models.cv3_speaker_sim import CV3SpeakerSim

        # This will only initialize the wrapper, not the actual process
        model = CV3SpeakerSim(
            model_id="damo/speech_eres2net_base_sv_zh-cn_3dspeaker_16k",
            local_model_dir="pretrained",
            device="cuda:0",
        )

        print("✓ Model wrapper initialized successfully")
        print(f"  Command args: {model.command_args}")
        return True

    except Exception as e:
        print(f"✗ Model initialization failed: {e}")
        return False


def test_inference_with_dummy_audio():
    """Test inference with dummy audio files."""
    print("\nTest 2: Inference with Dummy Audio")
    print("-" * 50)

    audio1_path = None
    audio2_path = None

    try:
        from audio_evals.models.cv3_speaker_sim import CV3SpeakerSim

        # Create dummy audio files
        print("Creating dummy audio files...")
        audio1_path = create_dummy_audio(duration=1.0, frequency=440)
        audio2_path = create_dummy_audio(duration=1.0, frequency=880)

        if audio1_path is None or audio2_path is None:
            print("✗ Failed to create dummy audio files")
            return False

        print(f"  Audio 1: {audio1_path}")
        print(f"  Audio 2: {audio2_path}")

        # Initialize model
        print("\nInitializing model...")
        model = CV3SpeakerSim(
            model_id="damo/speech_eres2net_base_sv_zh-cn_3dspeaker_16k",
            local_model_dir="pretrained",
            device="cuda:0",
        )

        # Prepare prompt
        prompt = {"audios": [audio1_path, audio2_path]}

        # Run inference
        print("Running inference...")
        similarity = model._inference(prompt)

        print(f"✓ Inference completed successfully")
        print(f"  Speaker similarity: {similarity:.6f}")
        print(f"  Range check: {-1.0 <= similarity <= 1.0}")

        # Validate result
        if not isinstance(similarity, (int, float)):
            print(f"✗ Invalid result type: {type(similarity)}")
            return False

        if not (-1.0 <= similarity <= 1.0):
            print(f"✗ Similarity out of range: {similarity}")
            return False

        return True

    except Exception as e:
        print(f"✗ Inference test failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    finally:
        # Cleanup
        if audio1_path and os.path.exists(audio1_path):
            os.unlink(audio1_path)
        if audio2_path and os.path.exists(audio2_path):
            os.unlink(audio2_path)


def test_error_handling():
    """Test error handling with invalid inputs."""
    print("\nTest 3: Error Handling")
    print("-" * 50)

    try:
        from audio_evals.models.cv3_speaker_sim import CV3SpeakerSim

        model = CV3SpeakerSim()

        # Test with wrong number of audios
        try:
            prompt = {"audios": ["/tmp/audio1.wav"]}  # Only 1 audio
            model._inference(prompt)
            print("✗ Should have raised an assertion error")
            return False
        except AssertionError as e:
            print(f"✓ Correctly raised AssertionError for wrong number of audios")
            print(f"  Error message: {e}")

        # Test with 3 audios
        try:
            prompt = {"audios": ["/tmp/a1.wav", "/tmp/a2.wav", "/tmp/a3.wav"]}
            model._inference(prompt)
            print("✗ Should have raised an assertion error")
            return False
        except AssertionError as e:
            print(f"✓ Correctly raised AssertionError for 3 audios")
            print(f"  Error message: {e}")

        return True

    except Exception as e:
        print(f"✗ Error handling test failed: {e}")
        return False


def test_model_variants():
    """Test initialization with different model variants."""
    print("\nTest 4: Model Variants")
    print("-" * 50)

    model_ids = [
        "damo/speech_campplus_sv_en_voxceleb_16k",
        "damo/speech_campplus_sv_zh-cn_16k-common",
        "damo/speech_eres2net_sv_en_voxceleb_16k",
        "damo/speech_eres2net_sv_zh-cn_16k-common",
        "damo/speech_eres2net_base_sv_zh-cn_3dspeaker_16k",
        "damo/speech_eres2net_large_sv_zh-cn_3dspeaker_16k",
    ]

    try:
        from audio_evals.models.cv3_speaker_sim import CV3SpeakerSim

        for model_id in model_ids:
            model = CV3SpeakerSim(model_id=model_id, local_model_dir="pretrained")
            print(f"✓ {model_id}")

        print(f"\n✓ All {len(model_ids)} model variants can be initialized")
        return True

    except Exception as e:
        print(f"✗ Model variant test failed: {e}")
        return False


def run_all_tests():
    """Run all integration tests."""
    print("=" * 60)
    print("CV3SpeakerSim Integration Tests")
    print("=" * 60)

    results = []

    # Test 1: Basic initialization
    results.append(("Model Initialization", test_model_initialization()))

    # Test 3: Error handling (doesn't require model files)
    results.append(("Error Handling", test_error_handling()))

    # Test 4: Model variants
    results.append(("Model Variants", test_model_variants()))

    # Test 2: Full inference (requires model files and 3D-Speaker library)
    print("\n" + "=" * 60)
    print("NOTE: The following test requires:")
    print("  1. 3D-Speaker library installed")
    print("  2. Model files downloaded in 'pretrained/' directory")
    print("  3. CUDA-capable GPU")
    print("=" * 60)

    response = input("\nRun full inference test? (y/n): ").strip().lower()
    if response == "y":
        results.append(("Full Inference", test_inference_with_dummy_audio()))
    else:
        print("Skipping full inference test")

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{name:.<40} {status}")

    print("=" * 60)
    print(f"Total: {passed}/{total} tests passed")
    print("=" * 60)

    return all(result for _, result in results)


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
