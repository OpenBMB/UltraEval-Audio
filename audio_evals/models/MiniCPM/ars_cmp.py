from audio_evals.models.model import Model
from typing import Dict
from audio_evals.base import PromptStruct


class PipelineAudioCPM(Model):
    def __init__(self, asr, prompt, llm, sample_params: Dict[str, any] = None):
        super().__init__(True, sample_params)
        from audio_evals.registry import registry

        self.asr = registry.get_model(asr)
        self.prompt = prompt
        self.llm = registry.get_model(llm)

    def _inference(self, prompt: PromptStruct, **kwargs) -> str:

        audio_file, text = "", ""
        for content in prompt:
            if content["role"] == "user":
                for line in content["contents"]:
                    if line["type"] == "audio":
                        audio_file = line["value"]
                        break

        text = self.asr.inference({"audio": audio_file})
        print(text)
        from audio_evals.registry import registry

        p = registry.get_prompt(self.prompt)
        prompt = p.load(query=text)
        return self.llm.inference(prompt)
