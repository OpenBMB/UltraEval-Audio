import json
import logging
from typing import Dict
from audio_evals.base import PromptStruct
from audio_evals.models.model import OfflineModel
from audio_evals.isolate import isolated
import select


logger = logging.getLogger(__name__)


@isolated("audio_evals/lib/VoxCPM2/main.py")
class VoxCPM2(OfflineModel):
    def __init__(
        self,
        path: str,
        zipenhancer_path: str = "",
        sample_params: Dict = None,
        *args,
        **kwargs,
    ):
        """
        VoxCPM TTS模型集成类

        Args:
            path: VoxCPM模型目录路径
            sample_params: 采样参数字典
        """
        self.command_args = {
            "path": path,
        }
        if zipenhancer_path:
            self.command_args["zipenhancer_path"] = zipenhancer_path
        super().__init__(is_chat=True, sample_params=sample_params)

    def _inference(self, prompt: PromptStruct, **kwargs):
        """
        执行VoxCPM推理

        Args:
            prompt: 包含推理参数的提示结构
            **kwargs: 额外的推理参数

        Returns:
            str: 生成的音频文件路径
        """
        import uuid

        uid = str(uuid.uuid4())
        prefix = f"{uid}->"
        prompt.update(kwargs)

        # 等待进程输入可用
        while True:
            _, wlist, _ = select.select([], [self.process.stdin], [], 60)
            if wlist:
                self.process.stdin.write(f"{prefix}{json.dumps(prompt)}\n")
                self.process.stdin.flush()
                logger.debug("Input sent to VoxCPM process")
                break

        # 等待并读取输出
        while True:
            rlist, _, _ = select.select(
                [self.process.stdout, self.process.stderr], [], [], 60
            )
            if not rlist:
                err_msg = "Read timeout after 60 seconds"
                logger.error(err_msg)
                raise RuntimeError(err_msg)

            try:
                for stream in rlist:
                    if stream == self.process.stdout:
                        result = self.process.stdout.readline().strip()
                        if not result:
                            continue
                        if result.startswith(prefix):
                            # 发送关闭信号
                            self.process.stdin.write("{}close\n".format(prefix))
                            self.process.stdin.flush()
                            # 返回生成的音频文件路径
                            return result[len(prefix) :]
                        elif result.startswith("Error:"):
                            raise RuntimeError("VoxCPM failed: {}".format(result))
                        else:
                            logger.info(result)
                    elif stream == self.process.stderr:
                        err = self.process.stderr.readline().strip()
                        if err:
                            logger.error(f"Process stderr: {err}")
            except BlockingIOError as e:
                logger.error(f"BlockingIOError occurred: {str(e)}")
