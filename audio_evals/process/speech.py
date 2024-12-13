import os.path
from audio_evals.process.base import Process


class Speech2text(Process):

    def __init__(self, whisper_model: str = "whisper",):
        from audio_evals.registry import registry
        self.model = registry.get_model(whisper_model)
        self.prompt = registry.get_prompt("whisper-asr")

    def __call__(self, answer: str) -> str:
        assert os.path.exists(answer), "must be a valid audio file, but got {}".format(answer)
        real_prompt = self.prompt.load(WavPath=answer)
        return self.model.inference(real_prompt)
