from kimia_infer.api.kimia import KimiAudio
import os
import soundfile as sf


if __name__ == "__main__":

    model = KimiAudio(
        model_path="moonshotai/Kimi-Audio-7B-Instruct",
        load_detokenizer=True,
    )

    sampling_params = {
        "audio_temperature": 0.8,
        "audio_top_k": 10,
        "text_temperature": 0.0,
        "text_top_k": 5,
        "audio_repetition_penalty": 1.0,
        "audio_repetition_window_size": 64,
        "text_repetition_penalty": 1.0,
        "text_repetition_window_size": 16,
    }

    messages = [
        {"role": "user", "message_type": "text", "content": "请将音频内容转换为文字。"},
        {
            "role": "user",
            "message_type": "audio",
            "content": "test_audios/asr_example.wav",
        },
    ]

    wav, text = model.generate(messages, **sampling_params, output_type="text")
    print(">>> output text: ", text)

    output_dir = "test_audios/output"
    os.makedirs(output_dir, exist_ok=True)
    # audio2audio
    messages = [
        {
            "role": "user",
            "message_type": "audio",
            "content": "test_audios/qa_example.wav",
        }
    ]

    wav, text = model.generate(messages, **sampling_params, output_type="both")
    sf.write(
        os.path.join(output_dir, "output.wav"),
        wav.detach().cpu().view(-1).numpy(),
        24000,
    )
    print(">>> output text: ", text)
