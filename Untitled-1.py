#!/usr/bin/env python3
import os
import argparse
import numpy as np
import torch
import soundfile as sf
from typing import Optional
import voxcpm
from datasets import load_dataset
from tqdm import tqdm


class VoxCPMInference:
    def __init__(self, model_dir: Optional[str] = None):
        """
        初始化 VoxCPM TTS 推理器

        Args:
            model_dir: 模型目录路径，如果为 None 则自动解析
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🚀 使用设备: {self.device}")

        self.model_dir = model_dir or self._resolve_model_dir()
        print(f"📁 模型目录: {self.model_dir}")

        self.model = None
        self._load_model()

    def _load_model(self):
        """加载 VoxCPM 模型"""
        try:
            print("🔄 正在加载模型...")
            self.model = voxcpm.VoxCPMModel.from_local(self.model_dir)
            print("✅ 模型加载成功!")
        except Exception as e:
            raise RuntimeError(f"❌ 模型加载失败: {e}")

    def generate_speech(
        self,
        text: str,
        prompt_wav_path: Optional[str] = None,
        prompt_text: Optional[str] = None,
        cfg_value: float = 2.0,
        timesteps: int = 10,
        max_len: int = 1000,
        min_len: int = 10,
        output_path: Optional[str] = None,
        auto_save: bool = True,
    ) -> tuple[int, np.ndarray]:
        """
        生成语音

        Args:
            text: 要合成的文本
            prompt_wav_path: 参考音频文件路径（可选）
            prompt_text: 参考音频对应的文本（可选）
            cfg_value: CFG 值，控制生成质量 (1.0-3.0)
            timesteps: 推理步数 (4-10)
            max_len: 最大长度
            min_len: 最小长度
            output_path: 输出音频文件路径（可选）
            auto_save: 如果为True且没有指定output_path，将自动生成文件名保存

        Returns:
            tuple: (采样率, 音频波形数据)
        """
        if not text or not text.strip():
            raise ValueError("❌ 请输入要合成的文本")

        text = text.strip()[:512]  # 限制文本长度

        print(f"🎵 正在生成语音: '{text[:60]}{'...' if len(text) > 60 else ''}'")

        # 生成音频
        wav = self.model.generate(
            target_text=text,
            prompt_text=prompt_text or "",
            prompt_wav_path=prompt_wav_path or "",
            min_len=min_len,
            max_len=max_len,
            inference_timesteps=timesteps,
            cfg_value=cfg_value,
        )

        # 转换为 numpy 数组
        if wav.dim() == 2:
            wav_np = wav.squeeze(0).numpy()
        else:
            wav_np = wav.numpy()

        sample_rate = self.model.sample_rate

        # 保存音频文件
        if output_path:
            self._save_audio(wav_np, sample_rate, output_path)
        return sample_rate, wav_np

    def _save_audio(self, wav_data: np.ndarray, sample_rate: int, output_path: str):
        """
        保存音频数据到文件

        Args:
            wav_data: 音频波形数据
            sample_rate: 采样率
            output_path: 输出文件路径
        """
        try:
            # 确保输出目录存在
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)

            # 规范化音频数据
            wav_data = np.clip(wav_data, -1.0, 1.0)

            # 保存音频文件
            sf.write(output_path, wav_data, sample_rate)

            # 验证文件是否成功保存
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                duration = len(wav_data) / sample_rate
                print(f"💾 音频已保存: {output_path}")
                print(f"   文件大小: {file_size / 1024:.1f} KB")
                print(f"   音频时长: {duration:.2f} 秒")
                print(f"   采样率: {sample_rate} Hz")
            else:
                print(f"⚠️  音频文件保存失败: {output_path}")

        except Exception as e:
            print(f"❌ 保存音频文件时出错: {e}")
            # 尝试备用保存方法
            try:
                import scipy.io.wavfile as wavfile

                # 转换为16位整数格式
                wav_int16 = (wav_data * 32767).astype(np.int16)
                wavfile.write(output_path, sample_rate, wav_int16)
                print(f"💾 使用备用方法保存音频: {output_path}")
            except Exception as e2:
                print(f"❌ 备用保存方法也失败: {e2}")
                raise RuntimeError(f"无法保存音频文件: {output_path}")


def main():
    dataset_name = "/data/shiqundong/UltraEval-Audio/bosonai_EmergentTTS-Eval"
    dataset_hf = load_dataset(dataset_name, split="train")
    model_dir = (
        "/share_data/data11005/shiqundong/model/VoxCPM/VoxCPM-0.5B-20250831-stable"
    )
    try:
        # 初始化推理器
        inference = VoxCPMInference(model_dir=model_dir)

        for i in tqdm(range(len(dataset_hf))):
            row = dataset_hf[i]
            sample_rate, wav_data = inference.generate_speech(
                text=row["text_to_synthesize"], output_path=f"output/{i}.wav"
            )
    except Exception as e:
        print(f"❌ 推理失败: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
