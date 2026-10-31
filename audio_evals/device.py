"""Device resolution shared by the CLI, model adapters and scoring models.

Resolution order (first match wins):

1. ``AUDIO_EVALS_DEVICE`` environment variable (set by ``--device`` on the CLI).
   It is inherited by the isolated worker processes, so one flag controls
   every model of a run.
2. the device explicitly requested by the caller (e.g. from a registry config)
3. ``cuda:0`` when CUDA is available, otherwise ``cpu``

Asking for a CUDA device through ``AUDIO_EVALS_DEVICE`` on a PyTorch build
without CUDA raises a clear ``RuntimeError`` instead of letting PyTorch fail
later with ``AssertionError: Torch not compiled with CUDA enabled``.
"""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

DEVICE_ENV_VAR = "AUDIO_EVALS_DEVICE"
AUTO_DEVICE_VALUES = ("", "auto")


def cuda_available() -> Optional[bool]:
    """Return whether PyTorch reports a usable CUDA device.

    Returns ``None`` when PyTorch is not importable in this interpreter (the
    host process of an isolated model may not ship torch at all), so callers
    can tell "no CUDA" apart from "cannot tell".
    """
    try:
        import torch
    except ImportError:
        return None
    try:
        return bool(torch.cuda.is_available())
    except Exception:  # pragma: no cover - backend specific failures
        return False


def resolve_device(
    requested: Optional[str] = None, default_cuda: str = "cuda:0"
) -> str:
    """Pick the device string a model should run on.

    Args:
        requested: device asked for by the caller (``"cpu"``, ``"cuda:1"``...).
            ``None``, ``""`` and ``"auto"`` mean "pick automatically".
        default_cuda: device returned by automatic selection when CUDA is
            available.

    Returns:
        A device string such as ``"cpu"`` or ``"cuda:0"``.

    Raises:
        RuntimeError: ``AUDIO_EVALS_DEVICE`` names a CUDA device but the
            installed PyTorch has no CUDA support.
    """
    env_device = os.environ.get(DEVICE_ENV_VAR, "").strip()
    if env_device and env_device not in AUTO_DEVICE_VALUES:
        if env_device.startswith("cuda") and cuda_available() is False:
            raise RuntimeError(
                f"{DEVICE_ENV_VAR}={env_device} asks for a CUDA device, but the "
                "installed PyTorch has no CUDA support. Use `--device cpu` "
                f"(or `export {DEVICE_ENV_VAR}=cpu`), or install a CUDA-enabled PyTorch."
            )
        return env_device

    if isinstance(requested, str):
        requested = requested.strip()
    if requested is not None and requested not in AUTO_DEVICE_VALUES:
        return str(requested)

    if cuda_available():
        return default_cuda
    logger.info("CUDA is not available, running on CPU")
    return "cpu"
