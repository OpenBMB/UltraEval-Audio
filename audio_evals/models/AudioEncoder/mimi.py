import json
import logging
import select
from audio_evals.base import PromptStruct
from audio_evals.models.model import OfflineModel
from audio_evals.isolate import isolated


logger = logging.getLogger(__name__)


@isolated("audio_evals/lib/mimi/main.py")
class MIMI(OfflineModel):
    def __init__(
        self,
        path: str,
        mono: bool = False,
        stereo: bool = False,
        stream: bool = False,
        sample_params=None,
        *args,
        **kwargs,
    ):
        path = self._download_model(path)
        self.command_args = {
            "path": path,
            "stream": stream,
            "mono": mono,
            "stereo": stereo,
        }
        super().__init__(is_chat=True, sample_params=sample_params)

    def _inference(self, prompt: PromptStruct, **kwargs) -> str:

        audio_path = prompt["audio"]
        input_x = {"audio": audio_path}
        input_x.update(kwargs)

        import uuid

        uid = str(uuid.uuid4())
        prefix = f"{uid}->"

        while True:
            # 等待 stdin 可写
            _, wlist, _ = select.select([], [self.process.stdin], [], 60)
            if not wlist:
                raise RuntimeError("Write timeout after 60 seconds")

            try:
                self.process.stdin.write(
                    f"{prefix}{json.dumps(input_x, ensure_ascii=False)}\n"
                )
                self.process.stdin.flush()
                break
            except BlockingIOError:
                continue

        while True:
            # 等待 stdout 或 stderr 可读
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
                            self.process.stdin.write("{}close\n".format(prefix))
                            self.process.stdin.flush()
                            return result[len(prefix) :]
                        elif result.startswith("Error:"):
                            raise RuntimeError("MIMI failed: {}".format(result))
                        else:
                            logger.info(result)
                    elif stream == self.process.stderr:
                        err = self.process.stderr.readline().strip()
                        if err:
                            logger.error(f"Process stderr: {err}")
                        continue
            except BlockingIOError as e:
                logger.error(f"BlockingIOError occurred: {str(e)}")
                continue
