import errno
import glob
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from audio_evals import env_setup

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CHILD_SCRIPT = """
import os, sys, time
sys.path.insert(0, {repo!r})
from audio_evals import env_setup

counter, env_path, requirements = sys.argv[1:4]


def fake_setup(env_path, requirements_path, pre_command, python_version):
    os.makedirs(os.path.join(env_path, "bin"), exist_ok=True)
    time.sleep(1)  # keep the lock long enough for the siblings to contend for it
    with open(counter, "a") as f:
        f.write("%d\\n" % os.getpid())


env_setup._run_setup = fake_setup
env_setup.ensure_env(env_path, requirements)
"""


@pytest.fixture(autouse=True)
def clean_state(monkeypatch):
    env_setup._prepared.clear()
    for var in (
        "AUDIO_EVALS_SKIP_ENV_SETUP",
        "AUDIO_EVALS_FORCE_ENV_SETUP",
        "AUDIO_EVALS_ENV_LOCK_TIMEOUT",
        "AUDIO_EVALS_ENV_LOCK_STALE_AFTER",
    ):
        monkeypatch.delenv(var, raising=False)
    yield
    env_setup._prepared.clear()


@pytest.fixture
def env(tmp_path):
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("tqdm\n")

    class Env:
        path = str(tmp_path / "envs" / "demo")
        req = str(requirements)
        req_file = requirements
        counter = tmp_path / "counter.log"

        @property
        def setup_count(self):
            if not self.counter.exists():
                return 0
            return len(self.counter.read_text().splitlines())

        @property
        def markers(self):
            return glob.glob(os.path.join(self.path, env_setup.MARKER_PREFIX + "*"))

    return Env()


def fake_setup_factory(counter, delay=0.0):
    def fake_setup(env_path, requirements_path, pre_command, python_version):
        os.makedirs(os.path.join(env_path, "bin"), exist_ok=True)
        time.sleep(delay)
        with open(counter, "a") as f:
            f.write(f"{os.getpid()}\n")

    return fake_setup


def test_repeated_calls_run_setup_once(env, monkeypatch):
    monkeypatch.setattr(env_setup, "_run_setup", fake_setup_factory(env.counter))

    env_setup.ensure_env(env.path, env.req)
    env_setup.ensure_env(env.path, env.req)
    env_setup._prepared.clear()  # a fresh process still sees the marker
    env_setup.ensure_env(env.path, env.req)

    assert env.setup_count == 1
    assert len(env.markers) == 1


def test_failed_setup_is_retried(env, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("install failed")

    monkeypatch.setattr(env_setup, "_run_setup", boom)
    with pytest.raises(RuntimeError):
        env_setup.ensure_env(env.path, env.req)
    assert env.markers == []

    monkeypatch.setattr(env_setup, "_run_setup", fake_setup_factory(env.counter))
    env_setup.ensure_env(env.path, env.req)
    assert env.setup_count == 1


def test_changed_requirements_trigger_new_setup(env, monkeypatch):
    monkeypatch.setattr(env_setup, "_run_setup", fake_setup_factory(env.counter))

    env_setup.ensure_env(env.path, env.req)
    env.req_file.write_text("tqdm\nnumpy\n")
    env_setup.ensure_env(env.path, env.req)

    assert env.setup_count == 2
    assert len(env.markers) == 2


def test_different_pre_command_tracked_separately(env, monkeypatch):
    monkeypatch.setattr(env_setup, "_run_setup", fake_setup_factory(env.counter))

    env_setup.ensure_env(env.path, env.req, pre_command="echo a")
    env_setup.ensure_env(env.path, env.req, pre_command="echo a")
    env_setup.ensure_env(env.path, env.req, pre_command="echo b")

    assert env.setup_count == 2
    assert len(env.markers) == 2


def test_concurrent_threads_run_setup_once(env, monkeypatch):
    monkeypatch.setattr(
        env_setup, "_run_setup", fake_setup_factory(env.counter, delay=0.3)
    )

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [
            pool.submit(env_setup.ensure_env, env.path, env.req) for _ in range(8)
        ]
        for future in futures:
            future.result()

    assert env.setup_count == 1
    assert len(env.markers) == 1


def test_concurrent_processes_run_setup_once(env, tmp_path):
    script = tmp_path / "child.py"
    script.write_text(CHILD_SCRIPT.format(repo=REPO_ROOT))

    procs = [
        subprocess.Popen(
            [sys.executable, str(script), str(env.counter), env.path, env.req],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(4)
    ]
    for proc in procs:
        _, stderr = proc.communicate(timeout=120)
        assert proc.returncode == 0, stderr

    assert env.setup_count == 1
    assert len(env.markers) == 1


def test_skip_flag_bypasses_setup(env, monkeypatch):
    monkeypatch.setattr(env_setup, "_run_setup", fake_setup_factory(env.counter))
    monkeypatch.setenv("AUDIO_EVALS_SKIP_ENV_SETUP", "1")

    env_setup.ensure_env(env.path, env.req)

    assert env.setup_count == 0
    assert not os.path.exists(env.path)


def test_force_flag_reruns_once_per_process(env, monkeypatch):
    monkeypatch.setattr(env_setup, "_run_setup", fake_setup_factory(env.counter))
    env_setup.ensure_env(env.path, env.req)

    monkeypatch.setenv("AUDIO_EVALS_FORCE_ENV_SETUP", "1")
    env_setup._prepared.clear()
    env_setup.ensure_env(env.path, env.req)
    env_setup.ensure_env(env.path, env.req)

    assert env.setup_count == 2
    assert len(env.markers) == 1


def test_lock_is_exclusive_without_flock(env, monkeypatch):
    """On filesystems where flock is unavailable the O_EXCL fallback still holds."""

    def unsupported(*args, **kwargs):
        raise OSError(errno.ENOLCK, "no locks available")

    monkeypatch.setattr(env_setup.fcntl, "flock", unsupported)

    concurrent = []
    inside = threading.Semaphore(0)
    holders = {"count": 0}
    lock = threading.Lock()

    def worker():
        with env_setup.env_lock(env.path, timeout=30):
            with lock:
                holders["count"] += 1
                concurrent.append(holders["count"])
            time.sleep(0.3)
            with lock:
                holders["count"] -= 1
            inside.release()

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
        assert not t.is_alive()

    assert concurrent == [1, 1, 1, 1]
    assert not os.path.exists(env_setup._lock_path(env.path) + ".excl")


def test_stale_fallback_lock_is_broken(env, monkeypatch):
    def unsupported(*args, **kwargs):
        raise OSError(errno.ENOLCK, "no locks available")

    monkeypatch.setattr(env_setup.fcntl, "flock", unsupported)

    lock_file = env_setup._lock_path(env.path) + ".excl"
    with open(lock_file, "w") as f:
        f.write("dead holder")
    os.utime(lock_file, (time.time() - 3600, time.time() - 3600))

    monkeypatch.setattr(env_setup, "_run_setup", fake_setup_factory(env.counter))
    monkeypatch.setenv("AUDIO_EVALS_ENV_LOCK_STALE_AFTER", "60")
    monkeypatch.setenv("AUDIO_EVALS_ENV_LOCK_TIMEOUT", "30")

    env_setup.ensure_env(env.path, env.req)

    assert env.setup_count == 1
