import atexit
import logging
import os
import subprocess
import sys
import tempfile
import time
from typing import Dict
from audio_evals.base import PromptStruct
from audio_evals.models.model import OfflineModel

logger = logging.getLogger(__name__)


def run_in_virtualenv(env_path, command):
    """Run a command inside a virtual environment."""
    if os.name == "nt":
        activate_cmd = f"{env_path}Scripts\\activate && {command}"
    else:
        act_path = os.path.join(env_path, "bin/activate")
        activate_cmd = f"source {act_path} && {command}"
    return subprocess.run(activate_cmd, shell=True, check=True, executable="/bin/bash")


class WavLM(OfflineModel):
    def __init__(
        self, path: str, env_path, requirements_path, sample_params: Dict = None
    ):
        super().__init__(is_chat=False, sample_params=sample_params)
        if not os.path.exists(env_path):
            res = subprocess.run(
                ["virtualenv", env_path],
            )
            if res.returncode != 0:
                raise RuntimeError(
                    "Failed to create virtual environment: {}".format(res.stderr)
                )
        result = run_in_virtualenv(env_path, f"pip install -r {requirements_path}")
        if result.returncode != 0:
            raise RuntimeError(
                "Dependency installation failed: {}".format(result.stderr)
            )

        script_path = "audio_evals/lib/simo/simo.py"
        if env_path.endswith("/"):
            env_path = env_path[:-1]
        command = f"source {env_path}/bin/activate &&export LD_LIBRARY_PATH={env_path}/lib/python3.10/site-packages/nvidia/nvjitlink/lib&&{env_path}/bin/python {script_path} --path '{path}'"
        print(f"Running command: {command}")
        self.process = subprocess.Popen(
            command,
            shell=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            executable="/bin/bash",
        )

        def cleanup():
            if self.process.poll() is None:
                self.process.terminate()  # 发送SIGTERM
                try:
                    self.process.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    self.process.kill()  # 强制终止

        atexit.register(cleanup)

    def _inference(self, prompt: PromptStruct, **kwargs) -> float:
        audio_paths = prompt["audios"]
        assert len(audio_paths) == 2, "wav lm must be used with two audio files."
        self.process.stdin.write(f"{','.join(audio_paths)}\n")
        self.process.stdin.flush()
        while True:
            result = self.process.stdout.readline().strip()
            if result.startswith("Result:"):
                return float(result[7:])
            elif result.startswith("Error:"):
                raise RuntimeError("wav lm failed: {}".format(result))
            elif result == "":
                raise RuntimeError("empty response from wav lm. need retry")
            else:
                logger.info(result)
