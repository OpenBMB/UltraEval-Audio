import logging
import os
import sys
import threading
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Dict

from audio_evals.base import PromptStruct
from audio_evals.utils import retry
from huggingface_hub import snapshot_download
from audio_evals.constants import DEFAULT_MODEL_PATH

# the str type for pre-train model, the list type for chat model

logger = logging.getLogger(__name__)


class Model(ABC):
    def __init__(self, is_chat: bool, sample_params: Dict[str, any] = None):
        self.is_chat = is_chat
        if sample_params is None:
            sample_params = {}
        self.sample_params = sample_params

    @abstractmethod
    def _inference(self, prompt: PromptStruct, **kwargs):
        raise NotImplementedError()

    def inference(self, prompt: PromptStruct, **kwargs) -> str:
        if isinstance(prompt, list) and not self.is_chat:
            raise ValueError("struct input not match pre-train model")
        if isinstance(prompt, str) and self.is_chat:
            prompt = [{"role": "user", "contents": [{"type": "text", "value": prompt}]}]
        sample_params = deepcopy(self.sample_params)
        sample_params.update(kwargs)
        logger.debug(f"sample_params: {sample_params}\nprompt: {prompt}")
        return self._inference(prompt, **sample_params)


class OfflineModel(Model, ABC):

    def __init__(self, is_chat: bool, sample_params: Dict[str, any] = None):
        super().__init__(is_chat, sample_params)
        self.lock = threading.Lock()

    @staticmethod
    def _download_model(repo_id: str, repo_type: str = None) -> str:
        """Download model from HuggingFace Hub if not exists locally.

        Args:
            repo_id: HuggingFace repository ID (e.g. "openbmb/MiniCPM-o-2_6")

        Returns:
            str: Local path where model is downloaded
        """
        try:
            logger = logging.getLogger(__name__)
            logger.info(f"Downloading model from HuggingFace Hub: {repo_id}")
            local_dir = snapshot_download(
                repo_id=repo_id,
                repo_type=repo_type,
                local_dir=os.path.join(DEFAULT_MODEL_PATH, repo_id),
                resume_download=True,
                local_dir_use_symlinks=False,
            )
            logger.info(f"Model downloaded to: {local_dir}")
            return local_dir
        except Exception as e:
            logger.error(
                f"Failed to download model: {e} from HuggingFace Hub; try to download from ModelScope"
            )
            return self._download_model_from_modelscope(repo_id, repo_type)

    def inference(self, prompt: PromptStruct, **kwargs) -> str:
        with self.lock:
            return super().inference(prompt, **kwargs)

    @staticmethod
    def _download_model_from_modelscope(repo_id: str, repo_type: str = None) -> str:
        """
        从 ModelScope 平台下载模型（如果本地不存在的话）。

        Args:
            repo_id: ModelScope 平台的模型标识（例如 "damo/nlp_bert_base" 或类似）
            repo_type: 可选，类似 "model"、"dataset" 等

        Returns:
            str: 本地模型目录路径
        """
        logger = logging.getLogger(__name__)
        try:
            logger.info(f"Downloading model from ModelScope: {repo_id}")

            # 目标本地目录：把 repo_id 当作子目录
            local_dir = os.path.join(DEFAULT_MODEL_PATH, repo_id)

            from modelscope.hub.snapshot_download import snapshot_download

            model_dir = snapshot_download(
                repo_id,
                local_dir=local_dir,
            )

            logger.info(f"Model downloaded to: {model_dir}")
            return model_dir

        except Exception as e:
            logger.error(
                f"Failed to download model from ModelScope: {e}", exc_info=True
            )
            sys.exit(1)


class APIModel(Model, ABC):

    @retry(max_retries=3)
    def inference(self, prompt: PromptStruct, **kwargs) -> str:
        return super().inference(prompt, **kwargs)
