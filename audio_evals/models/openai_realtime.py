import logging
from typing import Dict, Any
from audio_evals.models.model import APIModel
from audio_evals.base import PromptStruct, EarlyStop
import asyncio
import json
import os
import base64

import websockets
from pydub import AudioSegment
import io
import numpy as np
import soundfile as sf
from scipy.signal import resample

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_URL = os.getenv("OPENAI_URL", "api.openai.com")

logger = logging.getLogger(__name__)


def save_to_file(data, filename, mode="w"):
    """
    Save data to a file.

    :param data: The data to save. Can be string, bytes, or any serializable object.
    :param filename: The name of the file to save to.
    :param mode: The file mode. 'w' for text, 'wb' for binary.
    """
    try:
        if mode == "w":
            with open(filename, mode, encoding="utf-8") as file:
                if isinstance(data, str):
                    file.write(data)
                else:
                    json.dump(data, file, indent=2)
        elif mode == "wb":
            with open(filename, mode) as file:
                file.write(data)
        print(f"Data saved to {filename}")
    except Exception as e:
        print(f"Error saving to file: {e}")


async def send_event(websocket, event):
    await websocket.send(json.dumps(event))


def audio_to_item_create_event(audio_bytes: bytes) -> dict:
    # Load the audio file from the byte stream
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes))

    # Resample to 24kHz mono pcm16
    pcm_audio = audio.set_frame_rate(24000).set_channels(1).set_sample_width(2).raw_data

    # Encode to base64 string
    pcm_base64 = base64.b64encode(pcm_audio).decode()

    event = {
        "type": "conversation.item.create",
        "item": {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_audio", "audio": pcm_base64}],
        },
    }
    return event


async def handle_function_call(websocket, function_call):
    print("Function call:", function_call)
    # Here you would typically call the actual function and send the result back
    function_result_event = {
        "type": "conversation.item.create",
        "item": {
            "type": "function_call_output",
            "function_call_id": function_call["id"],
            "output": json.dumps(
                {"temperature": 22, "unit": "celsius", "description": "Partly cloudy"}
            ),
        },
    }
    await send_event(websocket, function_result_event)

    # Request another response after sending function result
    response_create_event = {
        "type": "response.create",
        "response": {
            "modalities": ["text", "audio"],
        },
    }
    await send_event(websocket, response_create_event)


async def handle_audio_response(audio_base64):
    audio_bytes = base64.b64decode(audio_base64)
    print(f"Received audio response of {len(audio_bytes)} bytes")
    # Save the audio response to a file
    save_to_file(audio_bytes, "assistant_response.wav", mode="wb")


def float_to_16bit_pcm(float32_array):
    """Convert Float32Array to 16-bit PCM."""
    int16_array = (float32_array * 32767).astype(np.int16)
    return int16_array.tobytes()


def base64_encode_audio(float32_array):
    """Convert Float32Array to base64-encoded PCM16 data."""
    pcm_data = float_to_16bit_pcm(float32_array)
    return base64.b64encode(pcm_data).decode()


def resample_audio(audio_data, original_sample_rate, target_sample_rate):
    number_of_samples = round(
        len(audio_data) * float(target_sample_rate) / original_sample_rate
    )
    resampled_audio = resample(audio_data, number_of_samples)
    return resampled_audio.astype(np.int16)


def get_audio_with_rate(audio_file_path):
    audio = AudioSegment.from_file(audio_file_path)

    # 获取音频数据和采样率
    audio_data = np.array(audio.get_array_of_samples(), dtype="int16")
    sample_rate = audio.frame_rate

    return audio_data, sample_rate


async def stream_audio_files(websocket, file_paths):
    """Stream audio files to the API."""
    for audio_file_path in file_paths:
        sample_rate = 24000
        duration_ms = 100
        samples_per_chunk = sample_rate * (duration_ms / 1000)
        bytes_per_sample = 2
        bytes_per_chunk = int(samples_per_chunk * bytes_per_sample)
        try:
            audio_data, original_sample_rate = get_audio_with_rate(audio_file_path)
        except Exception as e:
            logger.error(f"Error reading audio file: {e}")
            raise EarlyStop(f"reading audio file failed {e}")

        if original_sample_rate != sample_rate:
            audio_data = resample_audio(audio_data, original_sample_rate, sample_rate)

        audio_bytes = audio_data.tobytes()

        for i in range(0, len(audio_bytes), bytes_per_chunk):
            chunk = audio_bytes[i : i + bytes_per_chunk]
            if chunk:
                await send_event(
                    websocket,
                    {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(chunk).decode("utf-8"),
                    },
                )
    # Commit the audio buffer
    await send_event(websocket, {"type": "input_audio_buffer.commit"})


async def audio_inf(url, text, audio_file):
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "OpenAI-Beta": "realtime=v1",
    }

    async with websockets.connect(url, extra_headers=headers) as websocket:

        print("Connected to OpenAI Realtime API")

        # Set up the session
        session_update_event = {
            "type": "session.update",
            "session": {
                "modalities": ["text"],
                "instructions": text,
                "tools": [],
                "voice": "alloy",
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 200,
                },
            },
        }
        await send_event(websocket, session_update_event)
        audio_files = [audio_file]
        await stream_audio_files(websocket, audio_files)
        audio_buffer = bytearray()
        response_create_event = {
            "type": "response.create",
            "response": {
                "modalities": [
                    "text",
                ],
            },
        }
        await send_event(websocket, response_create_event)

        try:
            while True:
                message = await websocket.recv()
                event = json.loads(message)
                print("Received event:", event["type"])

                if event["type"] == "conversation.item.created":
                    if (
                        event["item"]["type"] == "message"
                        and event["item"]["role"] == "assistant"
                    ):
                        for content in event["item"]["content"]:
                            if content["type"] == "text":
                                logger.debug("Assistant:", content["text"])
                                save_to_file(content["text"], "assistant_response.txt")
                            elif content["type"] == "audio":
                                await handle_audio_response(content["audio"])
                    elif event["item"]["type"] == "function_call":
                        await handle_function_call(
                            websocket, event["item"]["function_call"]
                        )
                elif event["type"] == "response.text.done":
                    logger.debug("Assistant:", event["text"])
                    return event["text"]
                elif event["type"] == "error":
                    logger.warning("Error:", event["error"])
                elif event["type"] == "input_audio_buffer.speech_started":
                    logger.debug("Speech started")
                elif event["type"] == "input_audio_buffer.speech_stopped":
                    logger.debug("Speech stopped")
                elif event["type"] == "input_audio_buffer.committed":
                    logger.debug("Audio buffer committed")
                elif event["type"] == "response.audio.delta":
                    audio_content = base64.b64decode(event["delta"])
                    audio_buffer.extend(audio_content)
                    logger.debug(
                        f"🔵 Received {len(audio_content)} bytes, total buffer size: {len(audio_buffer)}"
                    )
                elif event["type"] == "response.audio.done":
                    # 创建WAV文件
                    with open("assistant_response.wav", "wb") as wav_file:
                        audio_array = np.frombuffer(audio_buffer, dtype=np.int16)
                        sf.write(wav_file, audio_array, samplerate=24000)
                    audio_buffer.clear()
                    logger.debug("🔵 AI finished speaking.")

        except websockets.exceptions.ConnectionClosed as e:
            logger.error(f"{e} Disconnected from OpenAI Realtime API")
            raise e


class GPT4oAudio(APIModel):
    def __init__(self, model_name: str, sample_params: Dict[str, Any] = None):
        super().__init__(True, sample_params)
        self.url = "wss://{}/v1/realtime?model={}".format(OPENAI_URL, model_name)
        assert "OPENAI_API_KEY" in os.environ, ValueError(
            "not found OPENAI_API_KEY in your ENV"
        )
        logger.info(f"OpenAI Realtime API URL: {self.url}")

    def _inference(self, prompt: PromptStruct, **kwargs) -> str:
        audio, text = None, None
        for line in prompt[0]["contents"]:
            if line["type"] == "audio":
                audio = line["value"]
            if line["type"] == "text":
                text = line["value"]
        assert os.path.exists(audio), EarlyStop(f"not found audio file: {audio}")
        return asyncio.run(audio_inf(self.url, text, audio))
