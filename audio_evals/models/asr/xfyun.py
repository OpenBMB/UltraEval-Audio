import logging
import json
import os
import subprocess
import tempfile
import time
import base64
import urllib

import requests
import websocket
import _thread as thread
from typing import Dict
from datetime import datetime
from wsgiref.handlers import format_date_time
import hashlib
import hmac
from urllib.parse import urlencode

from pydub import AudioSegment

from audio_evals.base import PromptStruct
from audio_evals.models.model import Model
from audio_evals.models.openai_realtime import PYDUB_SUPPORTED_FORMATS

logger = logging.getLogger(__name__)

STATUS_FIRST_FRAME = 0  # 第一帧
STATUS_CONTINUE_FRAME = 1  # 中间帧
STATUS_LAST_FRAME = 2  # 最后一帧


class XFYunRealTime(Model):
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
        self.url = "wss://ws-api.xfyun.cn/v2/iat"
        self.result = ""

    def create_url(self):
        now = datetime.now()
        date = format_date_time(time.mktime(now.timetuple()))
        signature_origin = f"host: ws-api.xfyun.cn\ndate: {date}\nGET /v2/iat HTTP/1.1"
        signature_sha = hmac.new(
            self.api_secret.encode("utf-8"),
            signature_origin.encode("utf-8"),
            digestmod=hashlib.sha256,
        ).digest()
        signature_sha = base64.b64encode(signature_sha).decode("utf-8")
        authorization = base64.b64encode(
            f'api_key="{self.api_key}", algorithm="hmac-sha256", headers="host date request-line", signature="{signature_sha}"'.encode(
                "utf-8"
            )
        ).decode("utf-8")
        params = {
            "authorization": authorization,
            "date": date,
            "host": "ws-api.xfyun.cn",
        }
        return self.url + "?" + urlencode(params)

    def on_message(self, ws, message):
        try:
            print("Received message:", message)  # 先打印完整数据

            response = json.loads(message)
            if response["code"] != 0:
                logger.error(f"Error: {response['message']}, Code: {response['code']}")
            else:
                for item in response["data"]["result"]["ws"]:
                    for w in item["cw"]:
                        self.result += w["w"]
        except Exception as e:
            logger.error(f"Failed to parse response: {e}")

    def on_error(self, ws, error):
        logger.error(f"WebSocket Error: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        logger.info("WebSocket closed")

    def on_open(self, ws, audio_file):
        def run(*args):
            frame_size = 8000
            interval = 0.04
            status = STATUS_FIRST_FRAME

            with open(audio_file, "rb") as fp:
                while True:
                    buf = fp.read(frame_size)
                    if not buf:
                        status = STATUS_LAST_FRAME

                    frame = {
                        "data": {
                            "status": status,
                            "format": "audio/L16;rate=16000",
                            "audio": base64.b64encode(buf).decode("utf-8"),
                            "encoding": "raw",
                        }
                    }
                    if status == STATUS_FIRST_FRAME:
                        frame.update(
                            {
                                "common": {"app_id": self.app_id},
                                "business": {
                                    "domain": "iat",
                                    "language": "zh_cn",
                                    "accent": "mandarin",
                                },
                            }
                        )
                        status = STATUS_CONTINUE_FRAME

                    ws.send(json.dumps(frame))
                    if status == STATUS_LAST_FRAME:
                        time.sleep(1)
                        break
                    time.sleep(interval)
            ws.close()

        thread.start_new_thread(run, ())

    def _inference(self, prompt: PromptStruct, **kwargs) -> str:
        self.result = ""
        ws_url = self.create_url()
        ws = websocket.WebSocketApp(
            ws_url,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close,
        )
        audio_file_path = prompt["audio"]
        fd, temp_wav_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", audio_file_path, "-ar", "16000", temp_wav_path],
                capture_output=True,
                text=True,
                check=True,
            )
            ws.on_open = lambda ws: self.on_open(ws, temp_wav_path)
            ws.run_forever()

        finally:
            os.remove(temp_wav_path)
        return self.result


lfasr_host = "https://raasr.xfyun.cn/v2/api"
api_upload = "/upload"
api_get_result = "/getResult"


class IFlyASR(Model):
    def __init__(self, app_id: str, api_secret: str, sample_params: dict = None):
        super().__init__(True, sample_params)
        self.appid = app_id
        self.secret_key = api_secret
        self.ts = str(int(time.time()))
        self.signa = self.get_signa()

    def get_signa(self):
        """生成签名"""
        appid = self.appid
        secret_key = self.secret_key
        m2 = hashlib.md5()
        m2.update((appid + self.ts).encode("utf-8"))
        md5 = m2.hexdigest()
        md5 = bytes(md5, encoding="utf-8")
        # 以secret_key为key, 上面的md5为msg，使用hashlib.sha1加密结果为signa
        signa = hmac.new(secret_key.encode("utf-8"), md5, hashlib.sha1).digest()
        signa = base64.b64encode(signa)
        signa = str(signa, "utf-8")
        return signa

    def upload(self, upload_file_path, lang):
        """上传音频文件"""
        logger.info("开始上传音频文件")
        file_len = os.path.getsize(upload_file_path)
        file_name = os.path.basename(upload_file_path)

        param_dict = {
            "appId": self.appid,
            "signa": self.signa,
            "ts": self.ts,
            "fileSize": file_len,
            "fileName": file_name,
            "language": lang,
            "duration": "200",  # 设置音频时长（单位：秒），需要根据实际情况调整
        }
        data = open(upload_file_path, "rb").read(file_len)

        response = requests.post(
            url=lfasr_host + api_upload + "?" + urllib.parse.urlencode(param_dict),
            headers={"Content-type": "application/json"},
            data=data,
        )

        result = response.json()
        logger.debug(f"上传响应：{result}")
        return result

    def get_result(self, audio_path, lang):
        """获取音频转写结果"""
        uploadresp = self.upload(audio_path, lang)
        if "content" not in uploadresp:
            raise Exception(f"上传失败：{uploadresp}")
        orderId = uploadresp["content"]["orderId"]
        param_dict = {
            "appId": self.appid,
            "signa": self.signa,
            "ts": self.ts,
            "orderId": orderId,
        }

        logger.info("开始查询转写结果")
        status = 3  # 3: 未完成，4: 完成
        # 建议使用回调的方式查询结果，查询接口有请求频率限制
        while status == 3:
            response = requests.post(
                url=lfasr_host
                + api_get_result
                + "?"
                + urllib.parse.urlencode(param_dict),
                headers={"Content-type": "application/json"},
            )
            result = response.json()
            logger.debug(f"查询响应：{result}")
            status = result["content"]["orderInfo"]["status"]
            logger.debug(f"status={status}")
            if status == 4:
                break
            time.sleep(1)  # 每5秒查询一次

        logger.info("获取转写结果成功")
        return result

    @staticmethod
    def rt2sentence(rt):
        """rt转sentence"""
        return "".join([item["cw"][0]["w"] for item in rt["ws"]])

    @staticmethod
    def st2sentence(st):
        return " ".join([IFlyASR.rt2sentence(item) for item in st["rt"]])

    @staticmethod
    def lattice2sentce(lattice):
        vad = json.loads(lattice["json_1best"])
        return IFlyASR.st2sentence(vad["st"])

    @staticmethod
    def res2sentence(res):
        lattice = res["lattice"]
        return " ".join([IFlyASR.lattice2sentce(item) for item in lattice])

    def _inference(self, prompt: PromptStruct, **kwargs) -> str:
        """进行模型推理，返回转写文本"""
        # 处理输入的音频文件
        audio_path = prompt["audio"]
        with tempfile.NamedTemporaryFile(
            suffix=".wav",
        ) as wav_file:
            subprocess.run(
                ["ffmpeg", "-y", "-i", audio_path, "-ar", "16000", wav_file.name],
                capture_output=True,
                text=True,
                check=True,
            )
            # 获取转写结果
            result = self.get_result(audio_path, **kwargs)
            if result["code"] == "000000":
                # 提取识别结果
                try:
                    transcription = json.loads(result["content"]["orderResult"])
                except Exception as e:
                    raise Exception(f"解析响应失败：{result}")
                text = self.res2sentence(transcription)
                logger.info(f"识别结果：{text}")
                return text
            else:
                raise Exception(f"识别失败：{result['message']}")
