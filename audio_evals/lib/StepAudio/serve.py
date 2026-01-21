"""
Step-Audio-R1.1 vLLM server startup script.

This script:
1. Finds an available port
2. Starts the vLLM server
3. Outputs port info via stdout for the parent process
4. Keeps running until terminated
5. Properly cleans up vLLM subprocess on exit

Usage:
    python serve.py --model_path /path/to/Step-Audio-R1.1

Reference: https://huggingface.co/stepfun-ai/Step-Audio-R1.1
"""

import argparse
import atexit
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import requests


# Global process reference for cleanup
_vllm_process = None


def cleanup_vllm():
    """Cleanup vLLM subprocess and its entire process group."""
    global _vllm_process
    if _vllm_process is not None and _vllm_process.poll() is None:
        print("[INFO] Cleaning up vLLM process group...", file=sys.stderr, flush=True)
        try:
            # Get process group ID
            pgid = os.getpgid(_vllm_process.pid)

            # Try graceful termination of entire process group first
            try:
                os.killpg(pgid, signal.SIGTERM)
            except ProcessLookupError:
                pass

            try:
                _vllm_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                # Force kill entire process group if graceful termination fails
                print(
                    "[INFO] Force killing vLLM process group...",
                    file=sys.stderr,
                    flush=True,
                )
                try:
                    os.killpg(pgid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                try:
                    _vllm_process.wait(timeout=5)
                except:
                    pass
        except Exception as e:
            print(f"[WARN] Error during cleanup: {e}", file=sys.stderr, flush=True)
            # Fallback to regular kill
            try:
                _vllm_process.kill()
            except:
                pass
        print("[INFO] vLLM process cleaned up", file=sys.stderr, flush=True)


def signal_handler(signum, frame):
    """Handle termination signals."""
    print(
        f"[INFO] Received signal {signum}, shutting down...",
        file=sys.stderr,
        flush=True,
    )
    cleanup_vllm()
    sys.exit(0)


def find_available_port(start_port: int = 9999, max_tries: int = 100) -> int:
    """Find an available port starting from start_port."""
    for port in range(start_port, start_port + max_tries):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("localhost", port))
                return port
        except OSError:
            continue
    raise RuntimeError(
        f"Could not find available port in range {start_port}-{start_port + max_tries}"
    )


def wait_for_server(port: int, timeout: int = 600) -> bool:
    """Wait for vLLM server to be ready."""
    health_url = f"http://localhost:{port}/health"
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            response = requests.get(health_url, timeout=5)
            if response.status_code == 200:
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(3)

    return False


def main():
    global _vllm_process

    # Register signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    # Register cleanup on exit
    atexit.register(cleanup_vllm)

    parser = argparse.ArgumentParser(description="Start Step-Audio-R1.1 vLLM server")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to Step-Audio-R1.1 model"
    )
    parser.add_argument(
        "--start_port",
        type=int,
        default=9999,
        help="Starting port to search from (default: 9999)",
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Server host (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=4,
        help="Tensor parallel size (default: 4)",
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=16384,
        help="Max model length (default: 16384)",
    )
    parser.add_argument(
        "--max_num_seqs",
        type=int,
        default=32,
        help="Max number of sequences (default: 32)",
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.85,
        help="GPU memory utilization (default: 0.85)",
    )
    parser.add_argument(
        "--startup_timeout",
        type=int,
        default=600,
        help="Server startup timeout in seconds (default: 600)",
    )
    args = parser.parse_args()

    # Find available port
    port = find_available_port(args.start_port)
    print(f"[INFO] Found available port: {port}", file=sys.stderr, flush=True)

    # Chat template for Step-Audio-R1.1
    chat_template = r"""{%- macro render_content(content) -%}{%- if content is string -%}{{- content.replace("<audio_patch>\n", "<audio_patch>") -}}{%- elif content is mapping -%}{{- content['value'] if 'value' in content else content['text'] -}}{%- elif content is iterable -%}{%- for item in content -%}{%- if item.type == 'text' -%}{{- item['value'] if 'value' in item else item['text'] -}}{%- elif item.type == 'audio' -%}<audio_patch>{%- endif -%}{%- endfor -%}{%- endif -%}{%- endmacro -%}{%- if tools -%}{{- '<|BOT|>system\n' -}}{%- if messages[0]['role'] == 'system' -%}{{- render_content(messages[0]['content']) + '<|EOT|>' -}}{%- endif -%}{{- '<|BOT|>tool_json_schemas\n' + tools|tojson + '<|EOT|>' -}}{%- else -%}{%- if messages[0]['role'] == 'system' -%}{{- '<|BOT|>system\n' + render_content(messages[0]['content']) + '<|EOT|>' -}}{%- endif -%}{%- endif -%}{%- for message in messages -%}{%- if message["role"] == "user" -%}{{- '<|BOT|>human\n' + render_content(message["content"]) + '<|EOT|>' -}}{%- elif message["role"] == "assistant" -%}{{- '<|BOT|>assistant\n' + (render_content(message["content"]) if message["content"] else '') -}}{%- set is_last_assistant = true -%}{%- for m in messages[loop.index:] -%}{%- if m["role"] == "assistant" -%}{%- set is_last_assistant = false -%}{%- endif -%}{%- endfor -%}{%- if not is_last_assistant -%}{{- '<|EOT|>' -}}{%- endif -%}{%- elif message["role"] == "function_output" -%}{%- else -%}{%- if not (loop.first and message["role"] == "system") -%}{{- '<|BOT|>' + message["role"] + '\n' + render_content(message["content"]) + '<|EOT|>' -}}{%- endif -%}{%- endif -%}{%- endfor -%}{%- if add_generation_prompt -%}{{- '<|BOT|>assistant\n<think>\n' -}}{%- endif -%}"""

    # Build vLLM command
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        args.model_path,
        "--served-model-name",
        args.model_path.split("/")[-1],
        "--port",
        str(port),
        "--host",
        args.host,
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--trust-remote-code",
        "--enable-log-requests",
        "--interleave-mm-strings",
        "--chat-template",
        chat_template,
    ]

    print(f"[INFO] Starting vLLM server on port {port}...", file=sys.stderr, flush=True)

    # Start vLLM server - use process group so we can kill all child processes
    _vllm_process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        # Create new process group for easier cleanup
        preexec_fn=os.setsid,
    )

    # Thread to forward vLLM output to stderr
    def forward_output():
        try:
            for line in _vllm_process.stdout:
                print(f"[vLLM] {line.rstrip()}", file=sys.stderr, flush=True)
        except:
            pass

    output_thread = threading.Thread(target=forward_output, daemon=True)
    output_thread.start()

    # Wait for server to be ready
    print(
        f"[INFO] Waiting for server to be ready (timeout: {args.startup_timeout}s)...",
        file=sys.stderr,
        flush=True,
    )

    if wait_for_server(port, args.startup_timeout):
        # Server is ready! Output port info via stdout for parent process
        print(f"PORT:{port}", flush=True)
        print(f"[INFO] Server is ready on port {port}", file=sys.stderr, flush=True)

        # Keep running until process exits or is terminated
        try:
            _vllm_process.wait()
        except KeyboardInterrupt:
            pass
        finally:
            cleanup_vllm()
    else:
        print(
            f"[ERROR] Server failed to start within {args.startup_timeout} seconds",
            file=sys.stderr,
            flush=True,
        )
        cleanup_vllm()
        sys.exit(1)


if __name__ == "__main__":
    main()
