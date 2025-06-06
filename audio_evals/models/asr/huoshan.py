import asyncio
import logging
import subprocess
import tempfile
import time
import json
import uuid
import requests
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


class DoubaoOfflineASR(APIModel):
    def __init__(
        self,
        appid: str,
        token: str,
        sample_params: Dict[str, any] = None,
    ):
        super().__init__(True, sample_params)
        self.appid = appid
        self.token = token

    def _get_audio_info(self, audio_path: str) -> Dict[str, any]:
        """使用ffprobe获取音频文件信息"""
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=sample_rate,channels,bits_per_sample",
            "-of",
            "json",
            audio_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        info = json.loads(result.stdout)
        stream = info["streams"][0]

        return {
            "rate": int(stream["sample_rate"]),
            "channel": int(stream["channels"]),
            "bits": int(stream["bits_per_sample"]),
        }

    def _inference(self, prompt: PromptStruct, **kwargs) -> str:
        # 获取音频文件路径
        audio = prompt["audio"]

        try:
            # 获取音频文件信息
            audio_info = self._get_audio_info(audio)

            # 构建请求参数
            headers = {
                "X-Api-App-Key": self.appid,
                "X-Api-Access-Key": self.token,
                "X-Api-Resource-Id": "volc.bigasr.auc",
                "X-Api-Request-Id": str(uuid.uuid4()),
                "X-Api-Sequence": "-1",
            }

            # 构建请求体
            request = {
                "user": {"uid": "fake_uid"},
                "audio": {
                    "url": audio,
                    "format": kwargs.get("format", "wav"),
                    "codec": "raw",
                    "rate": audio_info["rate"],
                    "bits": audio_info["bits"],
                    "channel": audio_info["channel"],
                },
                "request": {
                    "model_name": "bigmodel",
                    "show_utterances": True,
                    "corpus": {"correct_table_name": "", "context": ""},
                },
            }

            # 提交任务
            submit_url = "https://openspeech.bytedance.com/api/v3/auc/bigmodel/submit"
            response = requests.post(
                submit_url, data=json.dumps(request), headers=headers
            )

            if (
                "X-Api-Status-Code" not in response.headers
                or response.headers["X-Api-Status-Code"] != "20000000"
            ):
                raise RuntimeError(f"Submit task failed: {response.headers}")

            task_id = headers["X-Api-Request-Id"]
            x_tt_logid = response.headers.get("X-Tt-Logid", "")

            # 查询任务结果
            query_url = "https://openspeech.bytedance.com/api/v3/auc/bigmodel/query"
            query_headers = headers.copy()
            query_headers["X-Tt-Logid"] = x_tt_logid

            while True:
                query_response = requests.post(
                    query_url, json.dumps({}), headers=query_headers
                )
                code = query_response.headers.get("X-Api-Status-Code", "")

                if code == "20000000":  # task finished
                    result = query_response.json()
                    if not result or "result" not in result:
                        raise RuntimeError("No result in response")

                    return result["result"]["text"]
                elif code != "20000001" and code != "20000002":  # task failed
                    raise RuntimeError(f"Query task failed: {query_response.headers}")

                time.sleep(1)

        except Exception as e:
            import traceback

            traceback.print_exc()
            logger.error(f"Simplex ASR error: {str(e)}")
            raise RuntimeError(f"ASR Error: {str(e)}")
