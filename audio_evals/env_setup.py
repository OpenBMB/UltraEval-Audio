"""Idempotent preparation of the virtualenvs used by ``@isolated`` models.

Several model instances (``IsolatedModelPool`` driven by ``--workers``) and
several independent eval processes may point at the same ``env_path``. Running
``uv pip install`` twice concurrently -- or while another instance is already
importing the site-packages of that venv -- corrupts the environment, so the
setup is guarded by three layers: an in-process cache, a fingerprint marker
file inside the venv, and a cross-process lock.
"""

import contextlib
import errno
import hashlib
import json
import logging
import os
import socket
import subprocess
import threading
import time

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX platforms
    fcntl = None

logger = logging.getLogger(__name__)

MARKER_PREFIX = ".audio_evals_ready_"
LOCK_DIR_NAME = ".locks"

DEFAULT_LOCK_TIMEOUT = 7200.0  # installing torch from scratch can take very long
DEFAULT_STALE_AFTER = 600.0
_HEARTBEAT_INTERVAL = 30.0
_INITIAL_POLL_INTERVAL = 0.1
_MAX_POLL_INTERVAL = 5.0
_WAIT_LOG_INTERVAL = 60.0

_prepared = set()
_prepared_lock = threading.Lock()


class _FlockUnsupported(Exception):
    """Raised when the filesystem does not implement flock (some NFS mounts)."""


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid %s=%r, falling back to %s", name, raw, default)
        return default


def fingerprint(
    requirements_path: str, pre_command: str = "", python_version: str = "3.10"
) -> str:
    """Identify one (requirements, pre_command, python) combination.

    Keeping the fingerprint in the marker file name lets a single venv be
    shared by several distinct requirement sets, and makes an edited
    requirements file trigger a reinstall automatically.
    """
    try:
        with open(requirements_path, "rb") as f:
            requirements = f.read().decode("utf-8", errors="replace")
    except OSError:
        requirements = ""
    payload = json.dumps(
        {
            "requirements_path": requirements_path,
            "requirements": requirements,
            "pre_command": pre_command,
            "python": python_version,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def marker_path(env_path: str, fp: str) -> str:
    return os.path.join(env_path, MARKER_PREFIX + fp)


def _lock_path(env_path: str) -> str:
    env_path = os.path.abspath(env_path)
    lock_dir = os.path.join(os.path.dirname(env_path), LOCK_DIR_NAME)
    os.makedirs(lock_dir, exist_ok=True)
    return os.path.join(lock_dir, os.path.basename(env_path) + ".lock")


def _holder_info() -> str:
    return json.dumps(
        {
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    )


def _wait_until(acquire, timeout: float, description: str) -> None:
    """Poll ``acquire`` until it succeeds, logging progress while waiting."""
    deadline = time.monotonic() + timeout
    last_log = None
    delay = _INITIAL_POLL_INTERVAL
    while not acquire():
        now = time.monotonic()
        if now >= deadline:
            raise TimeoutError(
                f"Timed out after {timeout:.0f}s waiting for {description}. "
                f"If no other process is preparing the environment, remove the "
                f"lock file manually and retry."
            )
        if last_log is None or now - last_log >= _WAIT_LOG_INTERVAL:
            logger.info("Waiting for %s ...", description)
            last_log = now
        time.sleep(min(delay, max(deadline - now, 0.01)))
        delay = min(delay * 2, _MAX_POLL_INTERVAL)


def _try_flock(fd: int) -> bool:
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except BlockingIOError:
        return False
    except OSError as e:
        if e.errno in (errno.ENOLCK, errno.EOPNOTSUPP, errno.ENOSYS, errno.EINVAL):
            raise _FlockUnsupported(str(e))
        raise


def _acquire_flock(path: str, timeout: float, description: str) -> int:
    """Return a file descriptor holding the exclusive lock on ``path``."""
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        _wait_until(lambda: _try_flock(fd), timeout, description)
    except BaseException:
        os.close(fd)
        raise
    os.truncate(fd, 0)
    os.write(fd, _holder_info().encode("utf-8"))
    return fd


def _drop_if_stale(path: str, stale_after: float) -> None:
    try:
        age = time.time() - os.stat(path).st_mtime
    except FileNotFoundError:
        return
    if age <= stale_after:
        return
    logger.warning(
        "Removing stale environment lock %s (untouched for %.0fs)", path, age
    )
    with contextlib.suppress(FileNotFoundError):
        os.unlink(path)


def _heartbeat(path: str, stop: threading.Event) -> None:
    """Refresh the lock mtime so waiters do not consider it stale."""
    while not stop.wait(_HEARTBEAT_INTERVAL):
        try:
            os.utime(path, None)
        except OSError:
            return


@contextlib.contextmanager
def _exclusive_create_lock(
    path: str, timeout: float, stale_after: float, description: str
):
    """flock-free lock relying on atomic ``O_CREAT | O_EXCL`` file creation."""
    state = {"fd": None}

    def acquire():
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            _drop_if_stale(path, stale_after)
            return False
        state["fd"] = fd
        os.write(fd, _holder_info().encode("utf-8"))
        return True

    _wait_until(acquire, timeout, description)
    stop = threading.Event()
    beat = threading.Thread(
        target=_heartbeat, args=(path, stop), daemon=True, name="env-lock-heartbeat"
    )
    beat.start()
    try:
        yield
    finally:
        stop.set()
        beat.join(timeout=1)
        os.close(state["fd"])
        with contextlib.suppress(FileNotFoundError):
            os.unlink(path)


@contextlib.contextmanager
def env_lock(env_path: str, timeout: float = None, stale_after: float = None):
    """Hold an exclusive cross-process lock while preparing ``env_path``."""
    timeout = (
        _float_env("AUDIO_EVALS_ENV_LOCK_TIMEOUT", DEFAULT_LOCK_TIMEOUT)
        if timeout is None
        else timeout
    )
    stale_after = (
        _float_env("AUDIO_EVALS_ENV_LOCK_STALE_AFTER", DEFAULT_STALE_AFTER)
        if stale_after is None
        else stale_after
    )
    path = _lock_path(env_path)
    description = f"the lock on {env_path} (held by another process preparing it)"

    fd = None
    if fcntl is not None:
        try:
            fd = _acquire_flock(path, timeout, description)
        except _FlockUnsupported as e:
            logger.warning(
                "flock is unavailable on %s (%s); falling back to an O_EXCL lock",
                path,
                e,
            )

    if fd is not None:
        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)
        return

    with _exclusive_create_lock(path + ".excl", timeout, stale_after, description):
        yield


def _run_setup(
    env_path: str, requirements_path: str, pre_command: str, python_version: str
) -> None:
    if not os.path.exists(os.path.join(env_path, "bin", "activate")):
        res = subprocess.run(
            ["uv", "venv", env_path, "--python", python_version, "--allow-existing"]
        )
        if res.returncode != 0:
            raise RuntimeError(
                f"Failed to create virtual environment at {env_path} "
                f"(exit code {res.returncode})"
            )

    # setuptools<81 is a workaround for the bug in uv pip install
    # UV_TORCH_BACKEND lets uv pick the torch CUDA build matching the
    # host GPU driver (defaults to auto, overridable from the env).
    subprocess.run(
        f"source {env_path}/bin/activate && "
        f'export UV_TORCH_BACKEND="${{UV_TORCH_BACKEND:-auto}}" && '
        f"{pre_command + '&& ' if pre_command else ''} uv pip install setuptools\\<81 && "
        f"uv pip install --index-strategy unsafe-best-match -r {requirements_path}",
        shell=True,
        check=True,
        executable="/bin/bash",
    )


def _write_marker(path: str, requirements_path: str, pre_command: str) -> None:
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(
            {
                "host": socket.gethostname(),
                "pid": os.getpid(),
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "requirements_path": requirements_path,
                "pre_command": pre_command,
            },
            f,
        )
    os.replace(tmp, path)


def ensure_env(
    env_path: str,
    requirements_path: str,
    pre_command: str = "",
    python_version: str = "3.10",
) -> str:
    """Make sure ``env_path`` holds a venv with ``requirements_path`` installed.

    Safe to call from several threads and several processes at once: the work
    happens at most once per (env_path, requirements, pre_command) combination,
    everybody else either returns immediately or waits for the running setup.
    """
    env_path = env_path.rstrip("/")
    if _env_flag("AUDIO_EVALS_SKIP_ENV_SETUP"):
        logger.info("AUDIO_EVALS_SKIP_ENV_SETUP is set, using %s as-is", env_path)
        return env_path

    force = _env_flag("AUDIO_EVALS_FORCE_ENV_SETUP")
    fp = fingerprint(requirements_path, pre_command, python_version)
    key = (os.path.abspath(env_path), fp)
    marker = marker_path(env_path, fp)

    with _prepared_lock:
        if key in _prepared:
            return env_path

    if not force and os.path.exists(marker):
        logger.info("Env %s already prepared (%s), skipping setup", env_path, fp)
        with _prepared_lock:
            _prepared.add(key)
        return env_path

    with env_lock(env_path):
        if not force and os.path.exists(marker):
            logger.info(
                "Env %s was prepared by another process (%s), skipping setup",
                env_path,
                fp,
            )
        else:
            logger.info(
                "Preparing env %s (%s) from %s", env_path, fp, requirements_path
            )
            start = time.monotonic()
            _run_setup(env_path, requirements_path, pre_command, python_version)
            _write_marker(marker, requirements_path, pre_command)
            logger.info("Env %s ready in %.1fs", env_path, time.monotonic() - start)

    with _prepared_lock:
        _prepared.add(key)
    return env_path
