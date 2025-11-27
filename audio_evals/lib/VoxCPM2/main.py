import argparse
import json
import select
import sys
import tempfile
import os
import numpy as np
import torch
import soundfile as sf
from typing import Optional
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from voxcpm.core import VoxCPM


class VoxCPMInference:
    def __init__(self, model_dir: str, zipenhancer_path=None):
        """
        初始化 VoxCPM TTS 推理器

        Args:
            model_dir: 模型目录路径
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")

        self.model_dir = model_dir
        logger.info(f"Model directory: {self.model_dir}")

        # 从环境变量获取 ENABLE_RTF 设置，默认为0
        self.enable_rtf = int(os.environ.get("ENABLE_RTF", "0"))
        logger.info(f"ENABLE_RTF: {self.enable_rtf}")

        self.model = None
        self.zipenhancer_path = zipenhancer_path or os.environ.get(
            "ZIPENHANCER_MODEL_PATH", None
        )
        self._load_model()

    def _load_model(self):
        """加载 VoxCPM 模型"""
        try:
            logger.info(
                f"Loading VoxCPM model... with voxcpm_model_path: {self.model_dir}, zipenhancer_path: {self.zipenhancer_path}, enable_denoiser: {True if self.zipenhancer_path else False}"
            )
            self.model = VoxCPM(
                voxcpm_model_path=self.model_dir,
                zipenhancer_model_path=self.zipenhancer_path,
                enable_denoiser=True if self.zipenhancer_path else False,
            )
            logger.info("VoxCPM model loaded successfully!")
        except Exception as e:
            raise RuntimeError(f"Failed to load VoxCPM model: {e}")

    def generate_speech(
        self,
        text: str,
        prompt_wav_path: Optional[str] = None,
        prompt_text: Optional[str] = None,
        cfg_value: float = 2.0,
        timesteps: int = 10,
        retry_badcase: bool = True,
        output_path: str = None,
        denoise: bool = True,
        normalize: bool = False,
    ):
        """
        生成语音并保存到临时文件

        Args:
            text: 要合成的文本
            prompt_wav_path: 参考音频文件路径（可选）
            prompt_text: 参考音频对应的文本（可选）
            cfg_value: CFG 值，控制生成质量 (1.0-3.0)
            timesteps: 推理步数 (4-10)
            max_len: 最大长度
            min_len: 最小长度
            output_path: 输出音频文件路径

        Returns:
            str or dict: 根据ENABLE_RTF设置返回音频文件路径或包含RTF信息的字典
        """
        if not text or not text.strip():
            raise ValueError("Please input text to synthesize")

        text = text.strip()[:512]  # 限制文本长度

        logger.info(
            f"Generating speech for: '{text[:60]}{'...' if len(text) > 60 else ''}'"
        )

        # 记录开始时间用于RTF计算
        start_time = time.time()

        logger.info(
            f"Generating speech for: '{text[:60]}{'...' if len(text) > 60 else ''}',"
            f"prompt_text: '{prompt_text[:60]}{'...' if len(prompt_text) > 60 else ''}',"
            f"prompt_wav_path: '{prompt_wav_path[:60]}{'...' if len(prompt_wav_path) > 60 else ''}',"
            f"cfg_value: {cfg_value},"
            f"inference_timesteps: {timesteps},"
            f"retry_badcase: {retry_badcase},"
            f"denoise: {denoise},"
            f"normalize: {normalize}"
        )

        # 生成音频
        wav = self.model.generate(
            text=text,
            prompt_text=prompt_text or "",
            prompt_wav_path=prompt_wav_path or "",
            inference_timesteps=timesteps,
            cfg_value=cfg_value,
            normalize=normalize,
            denoise=denoise,
            retry_badcase=retry_badcase,
        )

        # 记录结束时间
        end_time = time.time()
        inference_time = end_time - start_time

        # 保存音频文件
        if not output_path:
            output_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name

        sf.write(str(output_path), wav, 16000)

        # 根据ENABLE_RTF设置返回不同格式
        if self.enable_rtf == 1:
            # 计算音频时长
            audio_duration = len(wav) / 16000
            # 计算RTF (Real Time Factor)
            logger.info(
                f"audio_duration: {audio_duration}, inference_time: {inference_time}"
            )
            rtf = inference_time / audio_duration if audio_duration > 0 else 0

            result = {"audio": output_path, "RTF": rtf}
            logger.info(
                f"RTF: {rtf:.4f} (inference: {inference_time:.2f}s, audio: {audio_duration:.2f}s)"
            )
            return result
        else:
            return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to VoxCPM model directory",
    )
    parser.add_argument(
        "--zipenhancer_path",
        type=str,
        default="",
        required=False,
        help="Path to ZipEnhancer model directory",
    )

    config = parser.parse_args()
    model = VoxCPMInference(
        model_dir=config.path, zipenhancer_path=config.zipenhancer_path
    )

    logger.info(f"Using VoxCPM2 model")

    while True:
        try:
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
            x = json.loads(prompt[anchor + 2 :])

            # Extract parameters from input format
            text = x.get("text", "")
            prompt_audio = x.get("prompt_audio", "")
            prompt_text = x.get("prompt_text", "")
            cfg_value = x.get("cfg_value", 2.0)
            timesteps = x.get("timesteps", 10)
            retry_badcase = x.get("retry_badcase", True)
            denoise = x.get("denoise", False)
            normalize = x.get("normalize", False)

            # Generate speech
            result = model.generate_speech(
                text=text,
                prompt_wav_path=prompt_audio if prompt_audio else None,
                prompt_text=prompt_text if prompt_text else None,
                cfg_value=cfg_value,
                timesteps=timesteps,
                retry_badcase=retry_badcase,
                denoise=denoise,
                normalize=normalize,
            )

            retry = 3
            while retry:
                # 根据返回类型输出不同格式
                if isinstance(result, dict):
                    print(f"{prefix}{json.dumps(result)}", flush=True)
                else:
                    print(f"{prefix}{result}", flush=True)
                rlist, _, _ = select.select([sys.stdin], [], [], 1)
                if rlist:
                    finish = sys.stdin.readline().strip()
                    if finish == "{}close".format(prefix):
                        break
                print("not found close signal, will emit again", flush=True)
                retry -= 1
        except Exception as e:
            import traceback

            traceback.print_exc()
            print(f"Error: {str(e)}", flush=True)
