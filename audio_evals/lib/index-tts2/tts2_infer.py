import os
from indextts.infer_v2 import IndexTTS2


class TTS2Processor:
    def __init__(
        self,
        model_dir,
        config_path,
        use_fp16=False,
        use_cuda_kernel=False,
        use_deepspeed=False,
    ):
        self.tts = IndexTTS2(
            cfg_path=config_path,
            model_dir=model_dir,
            use_fp16=use_fp16,
            use_cuda_kernel=use_cuda_kernel,
            use_deepspeed=use_deepspeed,
        )

    def process_text(self, text, audio_prompt, output_path):
        self.tts.infer(
            spk_audio_prompt=audio_prompt,
            text=text,
            output_path=output_path,
            verbose=False,
        )
        return output_path
