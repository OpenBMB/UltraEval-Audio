import os
import numpy as np
import torch
import gradio as gr
import spaces
from typing import Optional, Tuple
from funasr import AutoModel
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from voxcpm import VoxCPM


class VoxCPMDemo:
    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🚀 Running on device: {self.device}")

        # ASR model for prompt text recognition
        self.asr_model_id = "iic/SenseVoiceSmall"
        self.asr_model: Optional[AutoModel] = AutoModel(
            model=self.asr_model_id,
            disable_update=True,
            log_level="DEBUG",
            device="cuda:0" if self.device == "cuda" else "cpu",
        )

        # TTS model (lazy init)
        self.voxcpm_model: Optional[VoxCPM] = None
        self.default_local_model_dir = "./models/VoxCPM"

    # ---------- Model helpers ----------
    def _resolve_model_dir(self) -> str:
        """
        Resolve model directory:
        1) Use local checkpoint directory if exists
        2) If HF_REPO_ID env is set, download into models/{repo}
        3) Fallback to 'models'
        """
        if os.path.isdir(self.default_local_model_dir):
            return self.default_local_model_dir

        repo_id = os.environ.get("HF_REPO_ID", "").strip()
        if len(repo_id) == 0:
            repo_id = "openbmb/VoxCPM"
        target_dir = os.path.join("models", repo_id.replace("/", "__"))
        if not os.path.isdir(target_dir):
            try:
                from huggingface_hub import snapshot_download  # type: ignore

                os.makedirs(target_dir, exist_ok=True)
                print(
                    f"Downloading model from HF repo '{repo_id}' to '{target_dir}' ..."
                )
                snapshot_download(
                    repo_id=repo_id, local_dir=target_dir, local_dir_use_symlinks=False
                )
            except Exception as e:
                print(f"Warning: HF download failed: {e}. Falling back to 'data'.")
                return "models"
        return target_dir

    def get_or_load_voxcpm(self) -> VoxCPM:
        if self.voxcpm_model is not None:
            return self.voxcpm_model
        print("Model not loaded, initializing...")
        model_dir = self._resolve_model_dir()
        print(f"Using model dir: {model_dir}")
        self.voxcpm_model = VoxCPM(voxcpm_model_path=model_dir)
        print("Model loaded successfully.")
        return self.voxcpm_model

    # ---------- Functional endpoints ----------
    def prompt_wav_recognition(self, prompt_wav: Optional[str]) -> str:
        if prompt_wav is None:
            return ""
        res = self.asr_model.generate(input=prompt_wav, language="auto", use_itn=True)
        text = res[0]["text"].split("|>")[-1]
        return text

    def generate_tts_audio(
        self,
        text_input: str,
        prompt_wav_path_input: Optional[str] = None,
        prompt_text_input: Optional[str] = None,
        cfg_value_input: float = 2.0,
        inference_timesteps_input: int = 10,
        do_normalize: bool = True,
        denoise: bool = True,
    ) -> Tuple[int, np.ndarray]:
        """
        Generate speech from text using VoxCPM; optional reference audio for voice style guidance.
        Returns (sample_rate, waveform_numpy)
        """
        current_model = self.get_or_load_voxcpm()

        text = (text_input or "").strip()
        if len(text) == 0:
            raise ValueError("Please input text to synthesize.")

        prompt_wav_path = prompt_wav_path_input if prompt_wav_path_input else None
        prompt_text = prompt_text_input if prompt_text_input else None

        print(f"Generating audio for text: '{text[:60]}...'")
        wav = current_model.generate(
            text=text,
            prompt_text=prompt_text,
            prompt_wav_path=prompt_wav_path,
            cfg_value=float(cfg_value_input),
            inference_timesteps=int(inference_timesteps_input),
            normalize=do_normalize,
            denoise=denoise,
        )
        return (16000, wav)


# ---------- UI Builders ----------


def create_demo_interface(demo: VoxCPMDemo):
    """Build the Gradio UI for VoxCPM demo."""
    # static assets (logo path)
    gr.set_static_paths(paths=[Path.cwd().absolute() / "assets"])

    with gr.Blocks(
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="gray",
            neutral_hue="slate",
            font=[gr.themes.GoogleFont("Inter"), "Arial", "sans-serif"],
        ),
        css="""
        .logo-container {
            text-align: center;
            margin: 0.5rem 0 1rem 0;
        }
        .logo-container img {
            height: 80px;
            width: auto;
            max-width: 200px;
            display: inline-block;
        }
        /* Hide summary text for our two accordions but keep the chevron/marker visible */
        #acc_quick details > summary,
        #acc_tips details > summary {
            color: transparent !important;
        }
        #acc_quick details > summary::marker,
        #acc_tips details > summary::marker {
            color: #1e293b !important; /* fallback text color for marker */
        }
        /* Bold labels for specific checkboxes */
        #chk_denoise label,
        #chk_denoise span,
        #chk_normalize label,
        #chk_normalize span {
            font-weight: 600;
        }
        """,
    ) as interface:
        # Header logo
        gr.HTML(
            '<div class="logo-container"><img src="/gradio_api/file=assets/logo_v2.jpeg" alt="VoxCPM Logo"></div>'
        )

        # Quick Start
        gr.Markdown("### **📋 Quick Start Guide**")
        with gr.Accordion(" ", open=False, elem_id="acc_quick"):
            gr.Markdown(
                """
            ### How to Use:
            1. **(Optional) Upload or record prompt Speech** - provides voice tone, style, and characteristics for synthesis
            2. **(Optional) Enter prompt text** - the text content of your audio (auto-recognition available)
            3. **Enter text to synthesize** - what you want the voice to say
            4. **Click Generate Speech** to create your audio
            """
            )

        # Pro Tips
        gr.Markdown("### **💡 Pro Tips**")
        with gr.Accordion(" ", open=False, elem_id="acc_tips"):
            gr.Markdown(
                """
            ### Audio Quality Settings:
            - **Enable "Prompt Speech Enhancement"** for clean studio-quality voice cloning
            - **Disable it** to preserve background atmosphere (cafe sounds, etc.)

            ### Text Input:
            - **Enable "Text Normalization"** for regular text (handles numbers, punctuation automatically)
            - **Disable it** for advanced phonetic control using {HH AH0 L OW1}(EN) format

            ### Long Text:
            - Break long content into paragraphs with empty lines for best flow

            ### Fine-tuning:
            - **CFG Scale**: Lower if voice sounds strained, higher for maximum text adherence
            - **Inference Timesteps**: Lower for speed, higher for quality
            """
            )

        # Main controls
        with gr.Row():
            with gr.Column():
                prompt_wav = gr.Audio(
                    sources=["upload", "microphone"],
                    type="filepath",
                    label="Prompt Speech",
                    value="./examples/example.wav",
                )
                DoDenoisePromptAudio = gr.Checkbox(
                    value=False,
                    label="Prompt Speech Enhancement",
                    elem_id="chk_denoise",
                )
                with gr.Row():
                    prompt_text = gr.Textbox(
                        value="Just by listening a few minutes a day, you'll be able to eliminate negative thoughts by conditioning your mind to be more positive.",
                        label="Prompt Text",
                        placeholder="Please enter the prompt text. Automatic recognition is supported, and you can correct the results yourself...",
                    )
                run_btn = gr.Button("Generate Speech", variant="primary")

            with gr.Column():
                cfg_value = gr.Slider(
                    minimum=1.0,
                    maximum=3.0,
                    value=2.0,
                    step=0.1,
                    label="CFG Value (Guidance Scale)",
                    info="Higher values increase adherence to prompt, lower values allow more creativity",
                )
                inference_timesteps = gr.Slider(
                    minimum=4,
                    maximum=30,
                    value=10,
                    step=1,
                    label="Inference Timesteps",
                    info="Number of inference timesteps for generation (higher values may improve quality but slower)",
                )
                with gr.Row():
                    text = gr.Textbox(
                        value="VoxCPM is an innovative end-to-end TTS model from ModelBest, designed to generate highly expressive speech.",
                        label="Text to synthesize",
                        info="By default, we use \\n to split text into sub-sentences, and a sub-sentence is treated as a single chunk to synthesize. Finally, all chunks are concatenated to form the final audio.",
                    )
                with gr.Row():
                    DoNormalizeText = gr.Checkbox(
                        value=False, label="Text Normalization", elem_id="chk_normalize"
                    )
                audio_output = gr.Audio(label="Output Audio")

        # Wiring
        run_btn.click(
            fn=demo.generate_tts_audio,
            inputs=[
                text,
                prompt_wav,
                prompt_text,
                cfg_value,
                inference_timesteps,
                DoNormalizeText,
                DoDenoisePromptAudio,
            ],
            outputs=[audio_output],
            show_progress=True,
            api_name="generate",
        )
        prompt_wav.change(
            fn=demo.prompt_wav_recognition, inputs=[prompt_wav], outputs=[prompt_text]
        )

    return interface


def run_demo(
    server_name: str = "localhost", server_port: int = 7860, show_error: bool = True
):
    demo = VoxCPMDemo()
    interface = create_demo_interface(demo)
    # Recommended to enable queue on Spaces for better throughput
    interface.queue(max_size=10).launch(
        server_name=server_name, server_port=server_port, show_error=show_error
    )


if __name__ == "__main__":
    run_demo(server_name="localhost", server_port=7860, show_error=True)
