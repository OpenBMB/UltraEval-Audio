import base64
import functools
import importlib
import logging
import os
import time
import typing

from audio_evals.base import EarlyStop

logger = logging.getLogger(__name__)


def retry(max_retries=3, default=None):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for _ in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except EarlyStop as e:
                    if default is not None:
                        return default
                    raise e
                except Exception as e:
                    last_exception = e
                    logger.error(f"retry after: {e}")
                    time.sleep(1)  # 可选：添加延迟
            if default is not None:
                return default
            raise last_exception  # 抛出最后一次捕获的异常

        return wrapper

    return decorator


def make_object(class_name: str, **kwargs: typing.Dict[str, typing.Any]) -> typing.Any:
    module_name, qualname = class_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    cls = getattr(module, qualname)
    return cls(**kwargs)


MIME_TYPE_MAP = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".ogg": "audio/ogg",
    ".flac": "audio/flac",
    ".aac": "audio/aac",
    ".m4a": "audio/mp4",
    ".opus": "audio/opus",
    ".webm": "audio/webm",
    # 可以根据需要添加更多文件格式的支持
}


def get_base64_from_file(file_path):
    with open(file_path, "rb") as file:
        file_content = file.read()
        base64_encoded = base64.b64encode(file_content).decode("utf-8")
        return base64_encoded


def convbase64(file_path):
    """
    将语音文件转换为包含MIME类型的Base64编码数据URI。

    :param file_path: 语音文件的路径
    :return: 包含MIME类型的Base64编码数据URI
    """
    try:
        file_extension = os.path.splitext(file_path)[1].lower()
        mime_type = MIME_TYPE_MAP.get(file_extension)

        if not mime_type:
            raise ValueError(f"Unsupported file format: {file_extension}")
        base64_encoded = get_base64_from_file(file_path)
        data_uri = f"data:{mime_type};base64,{base64_encoded}"
        return data_uri
    except Exception as e:
        print(f"Error converting file to Base64: {e}")
        return None
