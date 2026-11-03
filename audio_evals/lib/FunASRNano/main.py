import argparse
import json
import select
import sys
import traceback


def pick_device(device):
    """Fall back to CPU when the requested CUDA device is unavailable."""
    try:
        import torch
    except ImportError:
        return device
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        print(f"CUDA is not available, running on cpu instead of {device}", flush=True)
        return "cpu"
    return device


def get_model(path, hub="hf", model_revision="main", device="cuda:0"):
    from funasr import AutoModel

    return AutoModel(
        model=path,
        hub=hub,
        model_revision=model_revision,
        device=device,
        trust_remote_code=True,
        disable_update=True,
        disable_pbar=True,
    )


def transcribe(model, audio, language=None):
    request = {"input": audio}
    if language:
        request["language"] = language

    results = model.generate(**request)
    if not results:
        raise RuntimeError("Fun-ASR-Nano returned no result")
    if not isinstance(results[0], dict) or "text" not in results[0]:
        raise RuntimeError("Fun-ASR-Nano result does not contain text")
    return " ".join(str(results[0]["text"]).split())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True)
    parser.add_argument("--hub", default="hf")
    parser.add_argument("--model_revision", default="main")
    parser.add_argument("--device", default="cuda:0")
    config = parser.parse_args()

    model = get_model(
        path=config.path,
        hub=config.hub,
        model_revision=config.model_revision,
        device=pick_device(config.device),
    )
    print(f"Model loaded from checkpoint: {config.path}", flush=True)

    for raw_prompt in sys.stdin:
        raw_prompt = raw_prompt.rstrip("\n")
        anchor = raw_prompt.find("->")
        if anchor == -1:
            print("Error: Invalid request format; expected '<id>-><json>'", flush=True)
            continue

        prefix = raw_prompt[:anchor].strip() + "->"
        try:
            request = json.loads(raw_prompt[anchor + 2 :].strip())
            text = transcribe(
                model,
                audio=request["audio"],
                language=request.get("language"),
            )
            print(f"{prefix}{text}", flush=True)
            readable, _, _ = select.select([sys.stdin], [], [], 1)
            if readable:
                acknowledgement = sys.stdin.readline().strip()
                if acknowledgement != f"{prefix}close":
                    print(
                        "Worker received an unexpected acknowledgement", file=sys.stderr
                    )
        except Exception as error:
            traceback.print_exc(file=sys.stderr)
            print(f"Error:{error}", flush=True)


if __name__ == "__main__":
    main()
