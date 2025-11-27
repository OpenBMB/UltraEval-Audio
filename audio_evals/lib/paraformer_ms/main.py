import argparse
import select
import subprocess
import sys
import tempfile

import soundfile
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess


def get_model(path, is_streaming=False):
    model_cfg = {
        # "vad_model": "fsmn-vad",
        # "punc_model": "ct-punc-c",
    }
    if is_streaming:
        model_cfg = {}
    print("Loading model from: {}".format(path))
    model = AutoModel(model=path, **model_cfg)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", type=str, required=True, help="Path to checkpoint file"
    )
    parser.add_argument("--chunk_size", type=int, default=0, help="Chunk size")
    config = parser.parse_args()
    is_streaming = config.path.endswith("streaming") or config.path.endswith("online")
    chunk_size = [0, 10, 5]  # [0, 10, 5] 600ms, [0, 8, 4] 480ms
    encoder_chunk_look_back = (
        4  # number of chunks to lookback for encoder self-attention
    )
    decoder_chunk_look_back = (
        1  # number of encoder chunks to lookback for decoder cross-attention
    )

    m = get_model(config.path, is_streaming=is_streaming)
    print("Model loaded from checkpoint: {}".format(config.path))

    while True:
        prompt = input()
        anchor = prompt.find("->")
        if anchor == -1:
            print(
                "Error: Invalid conversation format, must contains  ->, but {}".format(
                    prompt
                ),
                flush=True,
            )
            continue
        prefix = prompt[:anchor].strip() + "->"
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav") as wav_file:
                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-i",
                        prompt[len(prefix) :],
                        "-ar",
                        "16000",
                        "-ac",
                        "1",
                        "-f",
                        "wav",
                        wav_file.name,
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                if is_streaming:
                    speech, sample_rate = soundfile.read(wav_file.name)
                    chunk_stride = chunk_size[1] * 960  # 600ms

                    cache = {}
                    total_chunk_num = int(len((speech) - 1) / chunk_stride + 1)
                    transcription = ""
                    for i in range(total_chunk_num):
                        speech_chunk = speech[i * chunk_stride : (i + 1) * chunk_stride]
                        is_final = i == total_chunk_num - 1
                        res = m.generate(
                            input=speech_chunk,
                            cache=cache,
                            is_final=is_final,
                            chunk_size=chunk_size,
                            encoder_chunk_look_back=encoder_chunk_look_back,
                            decoder_chunk_look_back=decoder_chunk_look_back,
                        )
                        transcription += "".join([item["text"] for item in res])
                else:
                    if config.chunk_size > 0:
                        texts = []
                        audio, sr = soundfile.read(wav_file.name)
                        for start in range(0, len(audio), config.chunk_size * sr):
                            chunk = audio[start : start + config.chunk_size * sr]
                            res = m.generate(input=chunk, batch_size_s=300)
                            if len(res) > 0:
                                text = res[0]["text"]
                                texts.append(text.strip())
                        transcription = "".join(t for t in texts if t)
                    else:
                        transcription = m.generate(
                            input=wav_file.name, batch_size_s=300
                        )[0]["text"]
                retry = 3
                while retry:
                    retry -= 1
                    print(
                        "{}{}".format(
                            prefix, rich_transcription_postprocess(transcription)
                        )
                    )
                    rlist, _, _ = select.select([sys.stdin], [], [], 1)
                    if rlist:
                        finish = sys.stdin.readline().strip()
                        if finish == "{}close".format(prefix):
                            break
                    print("not found close signal, will emit again", flush=True)
        except Exception as e:
            import traceback

            traceback.print_exc()
            print("Error:{}".format(e))
