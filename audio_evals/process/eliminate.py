from audio_evals.process.base import Process


class Eliminate(Process):

    def __init__(self, target: str):
        self.target = target

    def __call__(self, answer: str) -> str:
        return answer.replace(self.target, "")


class ForceStop(Process):
    def __init__(self, target: str):
        self.target = target

    def __call__(self, answer: str) -> str:
        return answer.split(self.target)[0]


class ExtractResponse(Process):
    def __init__(self, target: str):
        self.target = target

    def __call__(self, answer: str) -> str:
        return answer.split(self.target)[1]
