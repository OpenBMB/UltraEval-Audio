import openai
from pathlib import Path
import base64

# 1. 设置 OpenAI API Key
# 确保在系统环境变量中设置 OPENAI_API_KEY，
# 或直接用 openai.api_key = "YOUR_API_KEY"
openai.api_key = "53e6f9e4ea325e8a5923870a351835d0"
openai.api_base = "https://llmcenter.modelbest.co/llm/openai/v1"
# 2. 准备音频文件（wav/mp3/m4a 都行）
# 这是你录好的问题，比如 "input_q.wav"
audio_file_path = Path("BAC009S0764W0472.wav")

# 3. 调用 gpt-audio 模型进行 QA
#   模型名称可用 "gpt-4o-audio-preview" 支持语音输入
#   同时可返回文本回答，也可以附带语音回答
client = openai.OpenAI(base_url=openai.api_base, api_key=openai.api_key)

with open(audio_file_path, "rb") as f:
    audio_bytes = f.read()
audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

response = client.chat.completions.create(
    model="GEMINI_0g1iy4",  # 支持音频输入的模型
    modalities=["text", "audio"],  # 请求同时返回文本和音频
    # audio={"voice": "alloy", "format": "mp3"}, # 可选: 输出音频设定
    messages=[
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are a helpful assistant that answers questions from audio input.",
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "input_audio",
                    "input_audio": {"data": audio_b64, "format": "wav"},
                }
            ],
        },
    ],
)
print(response)
