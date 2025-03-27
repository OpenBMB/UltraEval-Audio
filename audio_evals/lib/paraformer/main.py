import argparse
import subprocess
import tempfile

import soundfile
from funasr import AutoModel


def get_model(path, is_streaming=False):
    model_cfg = {
        "vad_model": "fsmn-vad",
        "vad_model_revision": "v2.0.4",
        "punc_model": "ct-punc-c",
        "punc_model_revision": "v2.0.4",
    }
    if is_streaming:
        model_cfg = {}
    model = AutoModel(model=path, model_revision="v2.0.4", **model_cfg)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", type=str, required=True, help="Path to checkpoint file"
    )
    config = parser.parse_args()
    is_streaming = config.path.endswith("streaming")
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
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav") as wav_file:
                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-i",
                        prompt,
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
                    wanted = ""
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
                        wanted += "".join([item["text"] for item in res])
                    print("Result:{}".format(wanted))
                else:
                    res = m.generate(input=wav_file.name, batch_size_s=300)
                    print("Result:{}".format(res[0]["text"]))
        except Exception as e:
            import traceback

            traceback.print_exc()
            print("Error:{}".format(e))
