import atexit
import subprocess
import logging
from functools import wraps

from audio_evals.env_setup import ensure_env

logger = logging.getLogger(__name__)


def isolated(
    script_path: str, command_args_attr: str = "command_args", pre_command: str = ""
):
    def decorator(cls):
        original_init = cls.__init__

        @wraps(original_init)
        def new_init(self, env_path, requirements_path, *args, gpu_id=None, **kwargs):
            """
            Args:
                env_path: 虚拟环境路径
                requirements_path: 依赖文件路径
                gpu_id: 指定使用的 GPU ID，如 0, 1, 2。
                        如果为 None，则不设置 CUDA_VISIBLE_DEVICES（使用默认行为）
            """
            original_init(self, *args, **kwargs)
            if env_path.endswith("/"):
                env_path = env_path[:-1]

            # 保存 gpu_id 供外部查询
            self._gpu_id = gpu_id

            # 创建虚拟环境并安装依赖（跨线程/跨进程只会真正执行一次）
            ensure_env(env_path, requirements_path, pre_command)

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

            # uv-managed python-build-standalone interpreters sometimes keep their
            # shared libpythonX.Y under the underlying install's own lib/ dir,
            # which isn't on the default dynamic linker search path.
            python_base_prefix = subprocess.check_output(
                f"source {env_path}/bin/activate && python -c 'import sys; print(sys.base_prefix)'",
                shell=True,
                executable="/bin/bash",
                text=True,
            ).strip()
            python_lib_dir = f"{python_base_prefix}/lib"

            # 构建 LD_LIBRARY_PATH
            lib_path = (
                f"{env_path}/lib/python{major_minor}/site-packages/nvidia/nvjitlink/lib"
            )

            cuda_runtime_lib = f"{env_path}/lib/python{major_minor}/site-packages/nvidia/cuda_runtime/lib"

            # 构建命令行参数
            command_args = getattr(self, command_args_attr, {})
            args_str = " ".join(
                [
                    f"--{key} " if value == "" else f"--{key} '{value}'"
                    for key, value in command_args.items()
                ]
            )

            # 构建 CUDA_VISIBLE_DEVICES 设置
            cuda_env = ""
            if gpu_id is not None:
                # gpu_id 由 IsolatedModelPool 注入，仅用于隔离子进程可见的物理
                # GPU。子进程内设备会重新编号（首张卡仍是 cuda:0），因此模型
                # 注册配置不应再指定或根据 gpu_id 改写 device。
                cuda_env = f"export CUDA_VISIBLE_DEVICES={gpu_id} && "
                logger.info(
                    f"Setting CUDA_VISIBLE_DEVICES={gpu_id} for isolated process"
                )

            # 构建完整命令
            command = (
                f"source {env_path}/bin/activate && "
                f"{cuda_env}"
                f"export LD_LIBRARY_PATH={lib_path}:{cuda_runtime_lib}:{python_lib_dir}:$LD_LIBRARY_PATH && "
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
