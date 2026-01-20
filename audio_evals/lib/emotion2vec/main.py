"""
Interactive emotion recognition evaluation script using emotion2vec_plus_large.
This script runs in interactive mode, accepting audio files and predicting emotions.
"""

import sys
import select
import librosa
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

# Emotion labels used by emotion2vec_plus_large
EMOTION_LABELS = [
    "angry",
    "disgusted",
    "fearful",
    "happy",
    "neutral",
    "other",
    "sad",
    "surprised",
    "unk",
]


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Interactive emotion recognition evaluation."
    )
    parser.add_argument(
        "--model", default="iic/emotion2vec_plus_large", type=str, help="Model name"
    )
    parser.add_argument("--device", default="cuda:0", type=str, help="Device to use")

    args = parser.parse_args()

    print(f"Loading emotion2vec model: {args.model}", flush=True)

    # Initialize the emotion recognition pipeline
    inference_pipeline = pipeline(
        task=Tasks.emotion_recognition, model=args.model, device=args.device
    )

    print(f"Successfully loaded emotion2vec model", flush=True)
    print("Ready to process audio files. Format: <prefix>-><audio_path>", flush=True)

    # Interactive loop
    while True:
        try:
            prompt = input()

            # Parse the input
            anchor = prompt.find("->")
            if anchor == -1:
                print(
                    f"Error: Invalid format, must contain ->, but got: {prompt}",
                    flush=True,
                )
                continue

            prefix = prompt[:anchor].strip() + "->"
            audio_path = prompt[anchor + 2 :].strip()

            try:
                # Load audio at 16kHz
                y, sr = librosa.load(audio_path, sr=16000)

                # Run emotion recognition
                rec_result = inference_pipeline(
                    y, granularity="utterance", extract_embedding=False
                )
                scores = rec_result[0]["scores"]
                predicted_emotion = EMOTION_LABELS[scores.index(max(scores))]
                max_score = max(scores)

                # Send result with prefix: predicted_emotion,confidence_score
                retry = 3
                while retry:
                    print(f"{prefix}{predicted_emotion},{max_score:.4f}", flush=True)
                    rlist, _, _ = select.select([sys.stdin], [], [], 1)
                    if rlist:
                        finish = sys.stdin.readline().strip()
                        if finish == f"{prefix}close":
                            break
                    print("not found close signal, will emit again", flush=True)
                    retry -= 1

            except Exception as e:
                print(f"Error: {str(e)}", flush=True)

        except EOFError:
            print("Received EOF, exiting...", flush=True)
            break
        except Exception as e:
            print(f"Error: {str(e)}", flush=True)


if __name__ == "__main__":
    main()
