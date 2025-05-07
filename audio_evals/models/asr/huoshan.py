import asyncio
import logging
import subprocess
import tempfile
import time
from typing import Dict

from audio_evals.base import PromptStruct
from audio_evals.lib.doubao.simplex_websocket_demo import submit_task, query_task
from audio_evals.models.model import APIModel
from audio_evals.lib.streaming_asr_demo import AsrWsClient, execute_one

logger = logging.getLogger(__name__)


class ByteDanceASRModel(APIModel):
    def __init__(
        self,
        appid: str,
        token: str,
        cluster: str,
        sample_params: Dict[str, any] = None,
    ):
        super().__init__(True, sample_params)
        self.appid = appid
        self.token = token
        self.cluster = cluster

    def _inference(self, prompt: PromptStruct, **kwargs) -> str:
        # 获取音频文件路径
        audio = prompt["audio"]

        try:
            # 构建请求参数
            with tempfile.NamedTemporaryFile(suffix=".wav") as wav_file:
                # 使用 ffmpeg 转换音频格式为 WAV
                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-i",
                        audio,
                        "-ar",
                        "16000",
                        "-ac",
                        "1",
                        wav_file.name,
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                )

                #
                result = asyncio.run(
                    AsrWsClient(
                        audio_path=wav_file.name,
                        cluster=self.cluster,
                        appid=self.appid,
                        token=self.token,
                        format="wav",  # 默认使用wav格式
                        **kwargs,
                    ).execute()
                )

                # 检查结果
                if "payload_msg" not in result:
                    raise RuntimeError("No payload message in response")

                if result["payload_msg"]["code"] != 1000:
                    raise RuntimeError(f"ASR Error: {result['payload_msg']['message']}")
                return result["payload_msg"]["result"][0]["text"]

        except Exception as e:
            logger.error(f"ByteDance ASR error: {str(e)}")
            raise RuntimeError(f"ASR Error: {str(e)}")


class DoubaoASR(APIModel):
    def __init__(
        self,
        appid: str,
        token: str,
        sample_params: Dict[str, any] = None,
    ):
        super().__init__(True, sample_params)
        self.appid = appid
        self.token = token

    def _inference(self, prompt: PromptStruct, **kwargs) -> str:
        # 获取音频文件路径
        audio = prompt["audio"]
        from audio_evals.lib.doubao.stream_asr import execute_one

        try:
            # 构建请求参数
            with tempfile.NamedTemporaryFile(suffix=".wav") as wav_file:
                # 使用 ffmpeg 转换音频格式为 WAV
                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-i",
                        audio,
                        "-ar",
                        "16000",
                        "-ac",
                        "1",
                        wav_file.name,
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                return execute_one(wav_file.name, **kwargs)

        except Exception as e:
            logger.error(f"ByteDance ASR error: {str(e)}")
            raise RuntimeError(f"ASR Error: {str(e)}")
