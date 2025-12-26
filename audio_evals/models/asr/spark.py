import logging
import json
import base64
import hashlib
import hmac
import ssl
import subprocess
import tempfile
import time
import websocket
import threading
from datetime import datetime
from urllib.parse import urlencode
from typing import Dict
from wsgiref.handlers import format_date_time

from audio_evals.base import PromptStruct
from audio_evals.models.model import APIModel

logger = logging.getLogger(__name__)

STATUS_FIRST_FRAME = 0  # 第一帧的标识
STATUS_CONTINUE_FRAME = 1  # 中间帧标识
STATUS_LAST_FRAME = 2  # 最后一帧的标识


class SparkASR(APIModel):
    """讯飞星火大模型 ASR 实现"""

    def __init__(
        self,
        app_id: str,
        api_key: str,
        api_secret: str,
        sample_params: Dict[str, any] = None,
    ):
        super().__init__(True, sample_params)
        self.app_id = app_id
        self.api_key = api_key
        self.api_secret = api_secret
        self.url = "wss://iat.cn-huabei-1.xf-yun.com/v1"
        self.result = ""
        self._lock = threading.Lock()
        self.iat_params = {
            "domain": "slm",
            "language": "zh_cn",
            "accent": "mulacc",
            "result": {"encoding": "utf8", "compress": "raw", "format": "json"},
        }

    def _create_url(self) -> str:
        """生成鉴权url"""
        now = datetime.now()
        date = format_date_time(time.mktime(now.timetuple()))

        # 拼接字符串
        signature_origin = "host: " + "iat.cn-huabei-1.xf-yun.com" + "\n"
        signature_origin += "date: " + date + "\n"
        signature_origin += "GET " + "/v1 " + "HTTP/1.1"

        # 进行hmac-sha256进行加密
        signature_sha = hmac.new(
            self.api_secret.encode("utf-8"),
            signature_origin.encode("utf-8"),
            digestmod=hashlib.sha256,
        ).digest()
        signature_sha_base64 = base64.b64encode(signature_sha).decode("utf-8")

        authorization_origin = f'api_key="{self.api_key}", algorithm="hmac-sha256", headers="host date request-line", signature="{signature_sha_base64}"'
        authorization = base64.b64encode(authorization_origin.encode("utf-8")).decode(
            "utf-8"
        )

        # 将请求的鉴权参数组合为字典
        v = {
            "authorization": authorization,
            "date": date,
            "host": "iat.cn-huabei-1.xf-yun.com",
        }
        # 拼接鉴权参数，生成url
        url = self.url + "?" + urlencode(v)
        return url

    def _on_message(self, ws, message):
        """websocket消息回调"""
        try:
            message = json.loads(message)
            code = message["header"]["code"]
            status = message["header"]["status"]

            if code != 0:
                logger.error(f"请求错误：{code}")
                ws.close()
                return

            payload = message.get("payload")
            if payload:
                text = payload["result"]["text"]
                text = json.loads(str(base64.b64decode(text), "utf8"))
                text_ws = text["ws"]
                result = ""
                for i in text_ws:
                    for j in i["cw"]:
                        w = j["w"]
                        result += w
                with self._lock:
                    self.result = result

            if status == 2:
                ws.close()

        except Exception as e:
            logger.error(f"Failed to parse response: {e}")

    def _on_error(self, ws, error):
        """websocket错误回调"""
        logger.error(f"WebSocket Error: {error}")

    def _on_close(self, ws, close_status_code, close_msg):
        """websocket关闭回调"""
        logger.info("WebSocket closed")

    def _on_open(self, ws, audio_file):
        """websocket连接建立回调"""

        def run(*args):
            try:
                frame_size = 1280  # 每一帧的音频大小
                interval = 0.04  # 发送音频间隔(单位:s)
                status = STATUS_FIRST_FRAME  # 音频的状态信息
                with tempfile.NamedTemporaryFile(suffix=".wav") as wav_file:
                    subprocess.run(
                        [
                            "ffmpeg",
                            "-y",
                            "-i",
                            audio_file,
                            "-acodec",
                            "pcm_s16le",
                            "-ac",
                            "1",
                            "-ar",
                            "16000",
                            wav_file.name,
                        ],
                        capture_output=True,
                        text=True,
                        check=True,
                    )
                    with open(wav_file.name, "rb") as fp:
                        while True:
                            buf = fp.read(frame_size)
                            audio = str(base64.b64encode(buf), "utf-8")

                            # 文件结束
                            if not audio:
                                status = STATUS_LAST_FRAME

                            # 第一帧处理
                            if status == STATUS_FIRST_FRAME:
                                data = {
                                    "header": {"status": 0, "app_id": self.app_id},
                                    "parameter": {"iat": self.iat_params},
                                    "payload": {
                                        "audio": {
                                            "audio": audio,
                                            "sample_rate": 16000,
                                            "encoding": "raw",
                                        }
                                    },
                                }
                                ws.send(json.dumps(data))
                                status = STATUS_CONTINUE_FRAME

                            # 中间帧处理
                            elif status == STATUS_CONTINUE_FRAME:
                                data = {
                                    "header": {"status": 1, "app_id": self.app_id},
                                    "payload": {
                                        "audio": {
                                            "audio": audio,
                                            "sample_rate": 16000,
                                            "encoding": "raw",
                                        }
                                    },
                                }
                                ws.send(json.dumps(data))

                            # 最后一帧处理
                            elif status == STATUS_LAST_FRAME:
                                data = {
                                    "header": {"status": 2, "app_id": self.app_id},
                                    "payload": {
                                        "audio": {
                                            "audio": audio,
                                            "sample_rate": 16000,
                                            "encoding": "raw",
                                        }
                                    },
                                }
                                ws.send(json.dumps(data))
                                break

                            time.sleep(interval)

            except Exception as e:
                logger.error(f"Error in on_open: {e}")
                ws.close()

        threading.Thread(target=run).start()

    def _inference(self, prompt: PromptStruct, **kwargs) -> str:
        """实现推理接口"""
        self.result = ""
        ws_url = self._create_url()

        ws = websocket.WebSocketApp(
            ws_url,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )

        audio_file_path = prompt["audio"]
        ws.on_open = lambda ws: self._on_open(ws, audio_file_path)

        ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})
        return self.result
