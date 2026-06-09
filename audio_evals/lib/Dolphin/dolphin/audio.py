import logging
import numpy as np
import subprocess

from .constants import SAMPLE_RATE


logger = logging.getLogger("dolphin")


# copy from whisper
def load_audio(file: str, sr: int = SAMPLE_RATE):
    """
    Open an audio file and read as mono waveform, resampling as necessary.
    """

    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads",
        "0",
        "-i",
        file,
        "-f",
        "s16le",
        "-ac",
        "1",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sr),
        "-",
    ]

    try:
        out = subprocess.run(cmd, capture_output=True, check=True).stdout
    except Exception as e:
        logger.error(f"load audio error, {e}")
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


def convert_audio(inp: str, outp: str):
    try:
        cmd = [
            "ffmpeg",
            "-loglevel",
            "0",
            "-nostdin",
            "-threads",
            "0",
            "-i",
            inp,
            "-acodec",
            "pcm_s16le",
            "-ac",
            "1",
            "-ar",
            "16000",
            outp,
            "-y",
        ]
        subprocess.run(cmd, capture_output=False, check=True)
    except Exception as e:
        logger.error(f"convert audio error, {e}")
        raise RuntimeError(f"Failed to convert audio: {e.stderr.decode()}") from e
