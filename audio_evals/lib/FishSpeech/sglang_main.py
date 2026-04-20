# pyright: reportMissingImports=false
import argparse
import atexit
import logging
import os
import signal
import socket
import subprocess
import sys
import time

import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_server_process = None
LOCAL_HOST = "127.0.0.1"


def find_available_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((LOCAL_HOST, 0))
        return sock.getsockname()[1]


def cleanup_server():
    global _server_process
    if _server_process is None or _server_process.poll() is not None:
        return

    logger.info("Cleaning up FishAudio SGLang server process group")
    try:
        pgid = os.getpgid(_server_process.pid)
        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            pass

        try:
            _server_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            _server_process.wait(timeout=5)
    except Exception as exc:
        logger.warning("Failed to cleanup SGLang server gracefully: %s", exc)
        try:
            _server_process.kill()
        except Exception:
            pass


def signal_handler(signum, frame):
    logger.info("Received signal %s, shutting down...", signum)
    cleanup_server()
    sys.exit(0)


def wait_for_server(base_url: str, timeout: int = 1800) -> None:
    health_urls = [
        f"{base_url}/health",
        f"{base_url}/v1/health",
        f"{base_url}/v1/models",
    ]
    start = time.time()

    while time.time() - start < timeout:
        if _server_process is not None and _server_process.poll() is not None:
            raise RuntimeError(
                f"SGLang server exited early with code {_server_process.returncode}"
            )
        for url in health_urls:
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    logger.info("FishAudio SGLang server is ready via %s", url)
                    return
            except requests.RequestException:
                pass
        time.sleep(3)

    raise TimeoutError(f"SGLang server not ready within {timeout} seconds")


def start_server(args) -> str:
    global _server_process

    port = find_available_port()
    base_url = f"http://{LOCAL_HOST}:{port}"
    cmd = [
        sys.executable,
        "-m",
        "sglang_omni.cli.cli",
        "serve",
        "--model-path",
        args.path,
        "--config",
        args.config,
        "--host",
        LOCAL_HOST,
        "--port",
        str(port),
    ]

    logger.info("Starting FishAudio SGLang server: %s", " ".join(cmd))
    _server_process = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid,
    )

    wait_for_server(base_url, timeout=args.startup_timeout)
    return base_url


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True, help="Fish Audio S2 path")
    parser.add_argument(
        "--config",
        type=str,
        default="examples/configs/s2pro_tts.yaml",
        help="sglang-omni config path",
    )
    parser.add_argument("--startup_timeout", type=int, default=1800)
    args = parser.parse_args()

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    atexit.register(cleanup_server)

    try:
        base_url = start_server(args)
    except Exception as exc:
        print(
            f"Failed to initialize FishAudio SGLang server: {exc}",
            file=sys.stderr,
            flush=True,
        )
        raise

    print(f"PORT:{base_url}", flush=True)
    logger.info("FishSpeech sglang_main.py launched service at %s", base_url)

    try:
        _server_process.wait()
    finally:
        cleanup_server()
