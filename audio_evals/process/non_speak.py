import re

from audio_evals.process.base import Process


class O4VocalSoundExtractor(Process):

    def __init__(
        self,
    ):
        pass

    def __call__(self, answer: str) -> str:
        m = re.search(
            r"the speaker is in the state of\s+([a-zA-Z]+)", answer, re.IGNORECASE
        )
        print(m)
        if m:
            return m.group(1)
        else:
            raise ValueError(f"No match found for vocal sound state in {answer}")
