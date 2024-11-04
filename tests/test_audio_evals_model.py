import logging

from audio_evals.registry import registry

# 配置根日志记录器
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)


def test_gpt_audio():
    model = registry.get_model("gpt4o_audio")
    prompt = registry.get_prompt("asr")
    real_prompt = prompt.load(
        WavPath="/tmp/jfs-training-root/training-root/user/hf-download/AudioEvals/aishell-1/data_aishell/wav/test/S0764/BAC009S0764W0121.wav"
    )
    print(model.inference(real_prompt))
