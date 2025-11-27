import torch
import torchaudio
import os
import tempfile
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from huggingface_hub import snapshot_download
from .model.voxcpm import VoxCPMModel
from .utils.text_normalize import TextNormalizer


class VoxCPM:
    def __init__(
        self,
        voxcpm_model_path: str,
        zipenhancer_model_path: str = "iic/speech_zipenhancer_ans_multiloss_16k_base",
        enable_denoiser: bool = True,
    ):
        """Initialize VoxCPM TTS pipeline.

        Args:
            voxcpm_model_path: Local filesystem path to the VoxCPM model assets
                (weights, configs, etc.). Typically the directory returned by
                a prior download step.
            zipenhancer_model_path: ModelScope acoustic noise suppression model
                id or local path. If None, denoiser will not be initialized.
            enable_denoiser: Whether to initialize the denoiser pipeline.
        """
        print(
            f"voxcpm_model_path: {voxcpm_model_path}, zipenhancer_model_path: {zipenhancer_model_path}, enable_denoiser: {enable_denoiser}"
        )
        self.tts_model = VoxCPMModel.from_local(voxcpm_model_path)
        self.text_normalizer = TextNormalizer()
        if enable_denoiser and zipenhancer_model_path is not None:
            self.denoiser = pipeline(
                Tasks.acoustic_noise_suppression, model=zipenhancer_model_path
            )
        else:
            self.denoiser = None
        print("Warm up VoxCPMModel...")
        self.tts_model.generate(target_text="Hello, this is the first test sentence.")

    @classmethod
    def from_pretrained(
        cls,
        hf_model_id: str = "openbmb/VoxCPM",
        load_denoiser: bool = True,
        zipenhancer_model_id: str = "iic/speech_zipenhancer_ans_multiloss_16k_base",
        cache_dir: str = None,
        local_files_only: bool = False,
    ):
        """Instantiate ``VoxCPM`` from a Hugging Face Hub snapshot.

        Args:
            hf_model_id: Explicit Hugging Face repository id (e.g. "org/repo").
            load_denoiser: Whether to initialize the denoiser pipeline.
            zipenhancer_model_id: Denoiser model id or path for ModelScope
                acoustic noise suppression.
            cache_dir: Custom cache directory for the snapshot.
            local_files_only: If True, only use local files and do not attempt
                to download.

        Returns:
            VoxCPM: Initialized instance whose ``voxcpm_model_path`` points to
            the downloaded snapshot directory.

        Raises:
            ValueError: If neither a valid ``hf_model_id`` nor a resolvable
                ``hf_model_id`` is provided.
        """
        repo_id = hf_model_id
        if not repo_id:
            raise ValueError("You must provide hf_model_id")

        local_path = snapshot_download(
            repo_id=repo_id,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )

        return cls(
            voxcpm_model_path=local_path,
            zipenhancer_model_path=zipenhancer_model_id if load_denoiser else None,
            enable_denoiser=load_denoiser,
        )

    def _normalize_loudness(self, wav_path: str):
        audio, sr = torchaudio.load(wav_path)
        loudness = torchaudio.functional.loudness(audio, sr)
        normalized_audio = torchaudio.functional.gain(audio, -20 - loudness)
        torchaudio.save(wav_path, normalized_audio, sr)

    def generate(
        self,
        text: str,
        prompt_wav_path: str = None,
        prompt_text: str = None,
        cfg_value: float = 2.0,
        inference_timesteps: int = 10,
        max_length: int = 4096,
        normalize: bool = True,
        denoise: bool = True,
        retry_badcase: bool = True,
        retry_badcase_max_times: int = 3,
        retry_badcase_ratio_threshold: float = 6.0,
    ):
        """Synthesize speech for the given text and return a single waveform.

        This method optionally builds and reuses a prompt cache. If an external
        prompt (``prompt_wav_path`` + ``prompt_text``) is provided, it will be
        used for all sub-sentences. Otherwise, the prompt cache is built from
        the first generated result and reused for the remaining text chunks.

        Args:
            text: Input text. Can include newlines; each non-empty line is
                treated as a sub-sentence.
            prompt_wav_path: Path to a reference audio file for prompting.
            prompt_text: Text content corresponding to the prompt audio.
            cfg_value: Guidance scale for the generation model.
            inference_timesteps: Number of inference steps.
            max_length: Maximum token length during generation.
            normalize: Whether to run text normalization before generation.
            denoise: Whether to denoise the prompt audio if a denoiser is
                available.
            retry_badcase: Whether to retry badcase.
            retry_badcase_max_times: Maximum number of times to retry badcase.
            retry_badcase_ratio_threshold: Threshold for audio-to-text ratio.
        Returns:
            numpy.ndarray: 1D waveform array (float32) on CPU.
        """
        texts = text.split("\n")
        texts = [t.strip() for t in texts if t.strip()]
        final_wav = []
        temp_prompt_wav_path = None

        try:
            # if prompt_wav_path is not None and prompt_text is not None:
            #     if denoise and self.denoiser is not None:
            #         with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            #             temp_prompt_wav_path = tmp_file.name

            #         self.denoiser(prompt_wav_path, output_path=temp_prompt_wav_path)
            #         self._normalize_loudness(temp_prompt_wav_path)
            #         prompt_wav_path = temp_prompt_wav_path
            #     fixed_prompt_cache = self.tts_model.build_prompt_cache(
            #         prompt_wav_path=prompt_wav_path,
            #         prompt_text=prompt_text
            #     )
            # else:
            #     fixed_prompt_cache = None  # will be built from the first inference

            for sub_text in texts:
                if sub_text.strip() == "":
                    continue
                print("sub_text:", sub_text)
                if normalize:
                    sub_text = self.text_normalizer.normalize(sub_text)
                print("should be here")
                wav = self.tts_model.generate(
                    target_text=sub_text,
                    prompt_wav_path=prompt_wav_path,
                    prompt_text=prompt_text,
                    # prompt_cache=fixed_prompt_cache,
                    min_len=10,
                    max_len=1000,
                    inference_timesteps=inference_timesteps,
                    cfg_value=cfg_value,
                    retry_badcase=retry_badcase,
                    retry_badcase_max_times=retry_badcase_max_times,
                    retry_badcase_ratio_threshold=retry_badcase_ratio_threshold,
                )
                # if fixed_prompt_cache is None:
                #     fixed_prompt_cache = self.tts_model.merge_prompt_cache(
                #         original_cache=None,
                #         new_text_token=target_text_token,
                #         new_audio_feat=generated_audio_feat
                #     )
                final_wav.append(wav)

            return torch.cat(final_wav, dim=1).squeeze(0).cpu().numpy()

        finally:
            if temp_prompt_wav_path and os.path.exists(temp_prompt_wav_path):
                try:
                    os.unlink(temp_prompt_wav_path)
                except OSError:
                    pass
