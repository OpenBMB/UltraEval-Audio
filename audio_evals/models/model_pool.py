"""
IsolatedModelPool: 管理多个隔离模型实例，支持多 GPU 并发推理

适用于使用 @isolated 装饰器的离线模型，通过 CUDA_VISIBLE_DEVICES 实现 GPU 隔离
"""
import os
import queue
import logging
from typing import Callable, Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def get_available_gpu_ids() -> List[int]:
    """
    获取可用 GPU 的 ID 列表
    
    优先使用 CUDA_VISIBLE_DEVICES 环境变量中指定的 GPU，
    如果未设置则通过 nvidia-smi 获取所有可用 GPU。
    
    Returns:
        可用 GPU ID 列表，如 [0, 1, 2, 3]
    """
    # 优先检查 CUDA_VISIBLE_DEVICES 环境变量
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if cuda_visible:
        try:
            gpu_ids = [int(x.strip()) for x in cuda_visible.split(',') if x.strip()]
            if gpu_ids:
                logger.info(f"Using GPUs from CUDA_VISIBLE_DEVICES: {gpu_ids}")
                return gpu_ids
        except ValueError:
            pass
    
    # 通过 nvidia-smi 获取所有可用 GPU
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index', '--format=csv,noheader'],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            gpu_ids = [int(x.strip()) for x in result.stdout.strip().split('\n') if x.strip()]
            if gpu_ids:
                logger.info(f"Detected GPUs via nvidia-smi: {gpu_ids}")
                return gpu_ids
    except Exception as e:
        logger.warning(f"Failed to detect GPUs via nvidia-smi: {e}")
    
    # 默认返回 GPU 0
    logger.warning("No GPUs detected, using default GPU 0")
    return [0]


class IsolatedModelPool:
    """
    隔离模型池：创建多个模型实例，支持并发推理
    
    适用于使用 @isolated 装饰器的离线模型。
    每个模型实例通过 gpu_id 参数指定使用的 GPU，
    isolated 装饰器会设置 CUDA_VISIBLE_DEVICES 环境变量实现隔离。
    
    GPU 分配策略：
    - 当 num_instances >= len(gpu_ids) 时：GPU 循环分配，多个实例可能共用同一个 GPU
    - 当 num_instances < len(gpu_ids) 时：每个实例分配多个 GPU（均匀分配）
    
    Example:
        ```python
        # 场景1: 4 个 GPU，8 个实例 → 每个 GPU 跑 2 个实例
        pool = IsolatedModelPool(
            model_factory=lambda **kw: registry.get_model("qwen3-tts", **kw),
            model_kwargs={"path": "/path/to/model"},
            gpu_ids=[0, 1, 2, 3],
            num_instances=8,
        )
        
        # 场景2: 8 个 GPU，2 个实例 → 每个实例分配 4 个 GPU
        # 实例 0 使用 GPU 0,1,2,3；实例 1 使用 GPU 4,5,6,7
        pool = IsolatedModelPool(
            model_factory=lambda **kw: registry.get_model("qwen3-tts", **kw),
            model_kwargs={"path": "/path/to/model"},
            gpu_ids=[0, 1, 2, 3, 4, 5, 6, 7],
            num_instances=2,
        )
        
        # 并发推理时，会自动从池中获取空闲实例
        result = pool.inference(prompt)
        ```
    """
    
    def __init__(
        self,
        model_factory: Callable[..., Any],
        model_kwargs: Dict[str, Any],
        gpu_ids: Optional[List[int]] = None,
        num_instances: Optional[int] = None,
    ):
        """
        Args:
            model_factory: 模型创建函数，接受 **kwargs 参数
            model_kwargs: 传给 model_factory 的基础参数
            gpu_ids: 可用 GPU ID 列表，如 [0, 1, 2, 3]。如果为 None，自动检测
            num_instances: 模型实例数量。如果为 None，默认等于 GPU 数量
        """
        if gpu_ids is None:
            gpu_ids = get_available_gpu_ids()
        
        if not gpu_ids:
            raise ValueError("gpu_ids cannot be empty")
        
        if num_instances is None:
            num_instances = len(gpu_ids)
        
        if num_instances < 1:
            raise ValueError(f"num_instances must be >= 1, got {num_instances}")
        
        self.gpu_ids = gpu_ids
        self.num_instances = num_instances
        self._pool: queue.Queue = queue.Queue()
        self._models = []
        
        logger.info(
            f"Creating IsolatedModelPool with {num_instances} instances on GPUs {gpu_ids}"
        )
        
        # 创建多个模型实例，分配 GPU
        # 计算每个实例的 GPU 分配
        gpu_assignments = self._compute_gpu_assignments(gpu_ids, num_instances)
        
        for i in range(num_instances):
            assigned_gpus = gpu_assignments[i]
            # gpu_id 可以是单个 int 或逗号分隔的字符串（多 GPU）
            if len(assigned_gpus) == 1:
                gpu_id = assigned_gpus[0]
            else:
                gpu_id = ','.join(map(str, assigned_gpus))
            
            kwargs = model_kwargs.copy()
            kwargs['gpu_id'] = gpu_id
            logger.info(f"Creating model instance {i} on GPU(s) {gpu_id}")
            try:
                model = model_factory(**kwargs)
                self._models.append(model)
                self._pool.put(model)
                logger.info(f"Model instance {i} on GPU(s) {gpu_id} created successfully")
            except Exception as e:
                logger.error(f"Failed to create model instance {i} on GPU(s) {gpu_id}: {e}")
                # 清理已创建的实例
                self._cleanup()
                raise
        
        logger.info(
            f"IsolatedModelPool initialized: {len(self._models)} instances on {len(gpu_ids)} GPUs"
        )
    
    @staticmethod
    def _compute_gpu_assignments(gpu_ids: List[int], num_instances: int) -> List[List[int]]:
        """
        计算每个实例的 GPU 分配
        
        - 当 num_instances >= len(gpu_ids) 时：GPU 循环分配，多个实例可能共用同一个 GPU
        - 当 num_instances < len(gpu_ids) 时：每个实例分配多个 GPU
        
        Args:
            gpu_ids: 可用 GPU ID 列表
            num_instances: 实例数量
            
        Returns:
            每个实例分配的 GPU ID 列表，如 [[0, 1], [2, 3]] 表示实例 0 用 GPU 0,1，实例 1 用 GPU 2,3
        """
        n_gpus = len(gpu_ids)
        
        if num_instances >= n_gpus:
            # GPU 数量不足，每个实例分配一个 GPU（循环使用）
            return [[gpu_ids[i % n_gpus]] for i in range(num_instances)]
        else:
            # GPU 数量充足，每个实例分配多个 GPU
            # 计算基础分配数和余数
            base_count = n_gpus // num_instances
            remainder = n_gpus % num_instances
            
            assignments = []
            gpu_idx = 0
            for i in range(num_instances):
                # 前 remainder 个实例多分配 1 个 GPU
                count = base_count + (1 if i < remainder else 0)
                assigned = gpu_ids[gpu_idx:gpu_idx + count]
                assignments.append(assigned)
                gpu_idx += count
            
            return assignments
    
    def _acquire(self, timeout: float = None):
        """
        获取一个空闲的模型实例
        
        Args:
            timeout: 超时时间（秒），None 表示无限等待
            
        Returns:
            空闲的模型实例
            
        Raises:
            queue.Empty: 超时未获取到实例
        """
        return self._pool.get(timeout=timeout)
    
    def _release(self, model):
        """归还模型实例到池中"""
        self._pool.put(model)
    
    def inference(self, prompt, **kwargs) -> str:
        """
        从池中获取模型实例进行推理，完成后自动归还
        
        Args:
            prompt: 输入 prompt
            **kwargs: 其他推理参数
            
        Returns:
            推理结果
        """
        model = self._acquire()
        try:
            return model.inference(prompt, **kwargs)
        finally:
            self._release(model)
    
    def _cleanup(self):
        """清理所有模型实例"""
        for model in self._models:
            try:
                # 调用模型自身的释放方法（如果有）
                if hasattr(model, 'release') and callable(model.release):
                    model.release()
                elif hasattr(model, 'unload') and callable(model.unload):
                    model.unload()
                
                # 处理子进程
                if hasattr(model, 'process') and model.process is not None and model.process.poll() is None:
                    model.process.terminate()
                    try:
                        model.process.wait(timeout=5)
                    except Exception:
                        model.process.kill()
            except Exception as e:
                logger.warning(f"Error cleaning up model: {e}")
        self._models.clear()
        
        # 清空队列
        while not self._pool.empty():
            try:
                self._pool.get_nowait()
            except queue.Empty:
                break
    
    def release(self):
        """
        释放所有模型实例并清理 GPU 显存
        
        调用此方法后，模型池将不可用，需要重新创建。
        """
        logger.info("Releasing IsolatedModelPool...")
        
        # 清理所有模型实例
        self._cleanup()
        self.num_instances = 0
        logger.info("IsolatedModelPool released.")
    
    def __del__(self):
        """析构时清理所有模型实例"""
        self._cleanup()
    
    def __len__(self):
        """返回池中的实例数量"""
        return self.num_instances
    
    @property
    def available_count(self) -> int:
        """返回当前空闲的实例数量"""
        return self._pool.qsize()
