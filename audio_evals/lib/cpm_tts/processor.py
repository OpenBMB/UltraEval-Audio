from typing import List, Optional, Literal, Tuple
import torch
import torchaudio


class MelSpectrogramFeatures(torch.nn.Module):
    def __init__(
        self,
        sample_rate=24000,
        n_fft=1024,
        hop_length=256,
        n_mels=100,
        padding: Literal["center", "same"] = "center",
    ):
        super().__init__()
        if padding not in ["center", "same"]:
            raise ValueError("Padding must be 'center' or 'same'.")
        self.padding = padding
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            center=padding == "center",
            power=1,
        )

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        """
        audio: Tensor([num_channels, num_samples])
        """
        return super().__call__(audio)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        audio: Tensor([num_channels, num_samples])
        """
        mel: torch.Tensor = self.mel_spec(audio)
        features = torch.log(torch.clip(mel, min=1e-5))
        return features


class ChatTTSProcessor:
    def __init__(self, text_tokenizer):
        self.audio_processor = MelSpectrogramFeatures()  # fixed, run on CPU
        self.text_tokenizer = text_tokenizer

    def __call__(self, text_list, audio_list):
        assert len(text_list) == len(audio_list)
        input_ids_varlen = []
        for text in text_list:  # 很遗憾没有并行
            input_ids_ = self.text_tokenizer.encode(
                text, return_tensors="pt", add_special_tokens=False
            )  # [1, seq_len]
            input_ids_ = input_ids_.squeeze(0)  # [seq_len]
            input_ids_varlen.append(input_ids_)

        audio_features_varlen = []
        for (
            audio
        ) in (
            audio_list
        ):  # 这个是因为ChatTTS作者给出的代码就没有并行化 因为并行化可能会引入padding 再加上本身运算量不大，可以放在cpu上跑。（builder里跑）
            assert audio.shape.__len__() == 1  # [seq_len]
            try:
                mel = self.audio_processor(audio)  # [100(num_mel_bins), seq_len_mel]
            except Exception as e:
                print(
                    "fuck! there is an error with audio waveform. If you use a dataset __getitem__, will skip and use next data as compensate, will not halt training."
                )
                raise e
            audio_features_varlen.append(mel)

        return {
            "tts_input_ids_varlen": [input_ids_varlen],  # 返回形状为List[List[Tensor]]
            "tts_input_features_varlen": [
                audio_features_varlen
            ],  # 返回形状为List[List[Tensor]]
        }


if __name__ == "__main__":
    from transformers import AutoTokenizer
    import torch

    text_tokenizer = AutoTokenizer.from_pretrained(
        "/home/jeeves/xubokai/ChatTTS/asset/tokenizer"
    )  # by default, it add special token, not good
    processor = ChatTTSProcessor(text_tokenizer=text_tokenizer)

    result1 = processor.text_tokenizer.encode(
        "Hello, how are you?", return_tensors="pt", add_special_tokens=False
    ).squeeze(0)
    print(result1)

    result2 = processor.text_tokenizer.encode("hey Hello, how are you? fine")
    print(result2)

    # In [29]: processor.text_tokenizer.encode('Hello, how are you?')
    # Out[29]: [101, 8701, 117, 9510, 8995, 8357, 136, 102]

    # In [30]: processor.text_tokenizer.encode('hey Hello, how are you? fine')
    # Out[30]: [101, 13153, 8701, 117, 9510, 8995, 8357, 136, 8533, 8354, 102]

    processor.text_tokenizer.encode(
        "hey Hello, how are you? fine", add_special_tokens=False, return_tensors="pt"
    )  # will get [1, seq_len]

    text_list = ["hey bro!", "fuck!"]

    audio_list = [
        torch.randn(20000, dtype=torch.float32),
        torch.randn(10000, dtype=torch.float32),
    ]

    a = processor(text_list=text_list, audio_list=audio_list)
