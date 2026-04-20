import json
import logging
import os
import select
import tempfile
import threading
import uuid
from typing import Dict, Optional

import requests
import soundfile as sf

from audio_evals.base import PromptStruct
from audio_evals.isolate import isolated
from audio_evals.models.model import OfflineModel

logger = logging.getLogger(__name__)


def _extract_payload(prompt: PromptStruct) -> Dict:
    if isinstance(prompt, dict):
        return dict(prompt)
    if isinstance(prompt, list):
        payload = {}
        for message in prompt:
            for item in message.get("contents", []):
                if item["type"] == "text":
                    payload[item.get("key", "text")] = item["value"]
                elif item["type"] == "audio":
                    payload[item.get("key", "prompt_audio")] = item["value"]
        return payload
    return {"text": str(prompt)}


def _request_via_isolated_process(process, payload: Dict, model_name: str) -> str:
    prefix = f"{uuid.uuid4()}->"

    while True:
        _, wlist, _ = select.select([], [process.stdin], [], 180)
        if not wlist:
            raise RuntimeError("Write timeout after 180 seconds")
        try:
            process.stdin.write(f"{prefix}{json.dumps(payload, ensure_ascii=False)}\n")
            process.stdin.flush()
            logger.debug("prompt written to %s stdin", model_name)
            break
        except BlockingIOError:
            continue

    while True:
        rlist, _, _ = select.select([process.stdout, process.stderr], [], [], 1800)
        if not rlist:
            err_msg = f"{model_name} read timeout after 1800 seconds"
            logger.error(err_msg)
            raise RuntimeError(err_msg)

        try:
            for stream in rlist:
                if stream == process.stdout:
                    result = process.stdout.readline().strip()
                    if not result:
                        continue
                    if result.startswith(prefix):
                        process.stdin.write(f"{prefix}close\n")
                        process.stdin.flush()
                        return result[len(prefix) :]
                    if result.startswith("Error:"):
                        raise RuntimeError(f"{model_name} failed: {result}")
                    logger.info(result)
                elif stream == process.stderr:
                    err = process.stderr.readline().strip()
                    if err:
                        logger.error("Process stderr: %s", err)
        except BlockingIOError as exc:
            logger.error("BlockingIOError occurred: %s", exc)
            continue


def _response_suffix(response_format: str) -> str:
    return {
        "wav": ".wav",
        "mp3": ".mp3",
        "flac": ".flac",
        "opus": ".opus",
        "pcm16": ".pcm",
    }.get(response_format, f".{response_format}" if response_format else ".wav")


@isolated(
    "audio_evals/lib/FishSpeech/main.py",
    pre_command=(
        "mkdir -p ./third_party && "
        "([ ! -d './third_party/fish-speech' ] && "
        "git clone https://github.com/fishaudio/fish-speech.git ./third_party/fish-speech"
        ") || true"
    ),
)
class FishAudioS2Pro(OfflineModel):
    """
    Fish Audio S2 Pro wrapper.

    Supports:
    - plain TTS with only `text`
    - voice cloning with `prompt_audio` + `prompt_text`
    """

    def __init__(
        self,
        path: str,
        compile: bool = False,
        sample_params: Optional[Dict] = None,
        *args,
        **kwargs,
    ):
        if not os.path.exists(path):
            path = self._download_model(path)

        self.command_args = {
            "path": path,
        }
        if compile:
            self.command_args["compile"] = ""

        super().__init__(is_chat=True, sample_params=sample_params)

    def _inference(self, prompt: PromptStruct, **kwargs) -> str:
        payload = _extract_payload(prompt)
        payload.update(kwargs)
        return _request_via_isolated_process(self.process, payload, "Fish Audio S2")


@isolated(
    "audio_evals/lib/FishSpeech/sglang_main.py",
    pre_command=(
        "mkdir -p ./third_party && "
        "([ ! -d './third_party/sglang-omni' ] && "
        "git clone https://github.com/sgl-project/sglang-omni.git ./third_party/sglang-omni"
        ") || true && "
        "uv pip install -e './third_party/sglang-omni[s2pro]'"
    ),
)
class FishAudioS2ProSGLang(OfflineModel):
    """
    Fish Audio S2 Pro wrapper backed by an isolated sglang-omni server process.
    """

    def __init__(
        self,
        path: str,
        startup_timeout: int = 1800,
        request_timeout: int = 1800,
        sample_params: Optional[Dict] = None,
        *args,
        **kwargs,
    ):
        if not os.path.exists(path):
            path = self._download_model(path)

        resolved_config = os.path.abspath(
            os.path.join(
                "third_party",
                "sglang-omni",
                "examples",
                "configs",
                "s2pro_tts.yaml",
            )
        )

        self.command_args = {
            "path": path,
            "config": resolved_config,
            "startup_timeout": startup_timeout,
        }
        self.request_timeout = request_timeout
        self.base_url = None
        self.speech_url = None
        self._stdout_thread = None
        self._stderr_thread = None
        super().__init__(is_chat=True, sample_params=sample_params)

    def _start_pipe_drain(self, stream_name: str, log_method):
        thread_attr = f"_{stream_name}_thread"
        if getattr(self, thread_attr) is not None:
            return

        def _drain():
            stream = getattr(self.process, stream_name)
            while True:
                try:
                    line = stream.readline()
                except Exception:
                    return
                if not line:
                    return
                log_method(
                    "Fish Audio S2 SGLang launcher %s: %s", stream_name, line.rstrip()
                )

        thread = threading.Thread(target=_drain, daemon=True)
        setattr(self, thread_attr, thread)
        thread.start()

    def _start_pipe_drains(self):
        self._start_pipe_drain("stdout", logger.debug)
        self._start_pipe_drain("stderr", logger.debug)

    def _ensure_server_ready(self):
        if self.speech_url is not None:
            return

        while True:
            rlist, _, _ = select.select(
                [self.process.stdout, self.process.stderr], [], [], 1800
            )
            if not rlist:
                raise RuntimeError(
                    "Fish Audio S2 SGLang startup timeout after 1800 seconds"
                )

            if self.process.poll() is not None:
                raise RuntimeError(
                    f"Fish Audio S2 SGLang launcher exited early with code {self.process.returncode}"
                )

            for stream in rlist:
                if stream == self.process.stdout:
                    line = self.process.stdout.readline().strip()
                    if not line:
                        continue
                    if line.startswith("PORT:"):
                        self.base_url = line[len("PORT:") :].strip().rstrip("/")
                        self.speech_url = f"{self.base_url}/v1/audio/speech"
                        self._start_pipe_drains()
                        logger.info(
                            "Fish Audio S2 SGLang is ready at %s", self.speech_url
                        )
                        return
                    logger.info(line)
                elif stream == self.process.stderr:
                    err = self.process.stderr.readline().strip()
                    if err:
                        logger.error("Process stderr: %s", err)

    def _build_request_payload(self, payload: Dict) -> tuple[Dict, str]:
        text = payload.pop("text", "").strip()
        if not text:
            raise ValueError("Fish Audio S2 SGLang requires `text` in the prompt.")

        prompt_audio = payload.pop("prompt_audio", None)
        prompt_text = payload.pop("prompt_text", None)
        response_format = str(payload.get("response_format", "wav")).lower()

        request_payload = {"input": text, **payload}
        if prompt_audio:
            if not prompt_text:
                raise ValueError(
                    "Fish Audio S2 SGLang voice cloning requires both `prompt_audio` and `prompt_text`."
                )
            request_payload["references"] = [
                {"audio_path": prompt_audio, "text": prompt_text}
            ]

        return request_payload, response_format

    def _inference(self, prompt: PromptStruct, **kwargs) -> str:
        self._ensure_server_ready()

        payload = _extract_payload(prompt)
        payload.update(kwargs)
        request_payload, response_format = self._build_request_payload(payload)

        start_time = None
        enable_rtf = int(os.environ.get("ENABLE_RTF", "0")) == 1
        if enable_rtf:
            import time

            start_time = time.time()

        response = requests.post(
            self.speech_url,
            json=request_payload,
            timeout=self.request_timeout,
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"Fish Audio S2 SGLang request failed: status={response.status_code}, body={response.text}"
            )

        with tempfile.NamedTemporaryFile(
            suffix=_response_suffix(response_format), delete=False
        ) as f:
            f.write(response.content)
            output_path = f.name

        if not enable_rtf:
            return output_path

        import time

        inference_time = time.time() - start_time
        audio_duration = 0.0
        try:
            info = sf.info(output_path)
            if info.samplerate > 0:
                audio_duration = info.frames / info.samplerate
        except Exception:
            logger.warning("Failed to compute audio duration for %s", output_path)
        rtf = inference_time / audio_duration if audio_duration > 0 else 0.0
        return json.dumps({"audio": output_path, "RTF": rtf}, ensure_ascii=False)
