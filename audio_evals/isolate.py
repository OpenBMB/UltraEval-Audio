import atexit
import os
import subprocess
import logging
from functools import wraps

logger = logging.getLogger(__name__)


def isolated(
    script_path: str, command_args_attr: str = "command_args", pre_command: str = ""
):
    def decorator(cls):
        original_init = cls.__init__

        @wraps(original_init)
        def new_init(self, env_path, requirements_path, *args, **kwargs):
            original_init(self, *args, **kwargs)
            if env_path.endswith("/"):
                env_path = env_path[:-1]

            # 创建虚拟环境
            if not os.path.exists(env_path):
                res = subprocess.run(["virtualenv", env_path])
                if res.returncode != 0:
                    raise RuntimeError(
                        f"Failed to create virtual environment: {res.stderr}"
                    )

            # 安装依赖
            result = subprocess.run(
                f"source {env_path}/bin/activate &&{pre_command + ' &&' if pre_command else ''} uv pip install -r {requirements_path}",
                shell=True,
                check=True,
                executable="/bin/bash",
            )
            if result.returncode != 0:
                raise RuntimeError(f"Dependency installation failed: {result.stderr}")

            # 自动检测 Python 版本
            python_version = (
                subprocess.check_output(
                    f"source {env_path}/bin/activate && python --version",
                    shell=True,
                    executable="/bin/bash",
                    text=True,
                )
                .strip()
                .split()[1]
            )
            major_minor = ".".join(python_version.split(".")[:2])

            # 构建 LD_LIBRARY_PATH
            lib_path = (
                f"{env_path}/lib/python{major_minor}/site-packages/nvidia/nvjitlink/lib"
            )

            # 构建命令行参数
            command_args = getattr(self, command_args_attr, {})
            args_str = " ".join(
                [
                    f"--{key} " if value == "" else f"--{key} '{value}'"
                    for key, value in command_args.items()
                ]
            )

            # 构建完整命令
            command = (
                f"source {env_path}/bin/activate && "
                f"export LD_LIBRARY_PATH={lib_path} && "
                f"{env_path}/bin/python -u {script_path} {args_str}"
            )
            logger.info(f"Running command: {command}")
            self.process = subprocess.Popen(
                command,
                shell=True,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                executable="/bin/bash",
            )

            # 等待10秒，检查进程是否退出
            try:
                exit_code = self.process.wait(timeout=10)
                # 进程在10秒内退出了，打印输出
                stdout, stderr = self.process.communicate()
                logger.info(f"Process exited with code: {exit_code}")
                if stdout:
                    logger.info(f"STDOUT:\n{stdout}")
                if stderr:
                    logger.error(f"STDERR:\n{stderr}")
            except subprocess.TimeoutExpired:
                # 进程在10秒内没有退出，继续正常运行
                logger.info("Process is still running after 10 seconds")

            # 注册清理函数
            def cleanup():
                if self.process.poll() is None:
                    self.process.terminate()
                    try:
                        self.process.wait(timeout=3600)
                    except subprocess.TimeoutExpired:
                        self.process.kill()

            atexit.register(cleanup)

        cls.__init__ = new_init
        return cls

    return decorator
