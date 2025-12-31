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
                res = subprocess.run(["uv", "venv", env_path, "--python", "3.10"])
                if res.returncode != 0:
                    raise RuntimeError(
                        f"Failed to create virtual environment: {res.stderr}"
                    )

            # 安装依赖
            result = subprocess.run(
                # setuptools<81 is a workaround for the bug in uv pip install
                f"source {env_path}/bin/activate &&{pre_command + '&& ' if pre_command else ''} uv pip install setuptools<81 && uv pip install -r {requirements_path}",
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

            # 添加检查进程状态并打印错误信息的方法
            def check_process_status(self_ref):
                """检查进程状态，如果进程已退出则打印所有输出信息"""
                if self_ref.process.poll() is not None:
                    exit_code = self_ref.process.returncode
                    logger.error(f"Process has exited with code: {exit_code}")
                    try:
                        # 读取剩余的输出
                        stdout, stderr = self_ref.process.communicate(timeout=5)
                        if stdout:
                            logger.error(f"Process STDOUT:\n{stdout}")
                        if stderr:
                            logger.error(f"Process STDERR:\n{stderr}")
                    except Exception as e:
                        logger.error(f"Failed to read process output: {e}")
                    return False
                return True

            self.check_process_status = lambda: check_process_status(self)

            # 注册清理函数
            def cleanup():
                if self.process.poll() is None:
                    self.process.terminate()
                    try:
                        self.process.wait(timeout=3600)
                    except subprocess.TimeoutExpired:
                        self.process.kill()
                else:
                    # 进程已退出，打印输出信息
                    exit_code = self.process.returncode
                    logger.info(f"Process already exited with code: {exit_code}")
                    try:
                        stdout, stderr = self.process.communicate(timeout=5)
                        if stdout:
                            logger.info(f"Final STDOUT:\n{stdout}")
                        if stderr:
                            logger.error(f"Final STDERR:\n{stderr}")
                    except Exception as e:
                        logger.warning(f"Could not read final output: {e}")

            atexit.register(cleanup)

        cls.__init__ = new_init
        return cls

    return decorator
