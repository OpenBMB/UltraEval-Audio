from dataclasses import dataclass

# @dataclass(repr=False, eq=False)
# class Path:
#     vocos_ckpt_path: str = "asset/Vocos.pt"
#     dvae_ckpt_path: str = "asset/DVAE_full.pt"
#     gpt_ckpt_path: str = "asset/GPT.pt"
#     decoder_ckpt_path: str = "asset/Decoder.pt"
#     tokenizer_path: str = "asset/tokenizer.pt"


@dataclass(repr=False, eq=False)
class Decoder:
    idim: int = 384
    odim: int = 384
    hidden: int = 512
    n_layer: int = 12
    bn_dim: int = 128


@dataclass(repr=False, eq=False)
class VQ:
    dim: int = 1024
    levels: tuple = (5, 5, 5, 5)
    G: int = 2
    R: int = 2


@dataclass(repr=False, eq=False)
class DVAE:
    encoder: Decoder = Decoder(
        idim=512,
        odim=1024,
        hidden=256,
        n_layer=12,
        bn_dim=128,
    )
    decoder: Decoder = Decoder(
        idim=512,
        odim=512,
        hidden=256,
        n_layer=12,
        bn_dim=128,
    )
    vq: VQ = VQ()


@dataclass(repr=False, eq=False)
class GPT:
    hidden_size: int = 768
    intermediate_size: int = 3072
    num_attention_heads: int = 12
    num_hidden_layers: int = 20
    use_cache: bool = False
    max_position_embeddings: int = 4096

    spk_emb_dim: int = 192
    spk_KL: bool = False
    num_audio_tokens: int = 626
    num_vq: int = 4


@dataclass(repr=False, eq=False)
class FeatureExtractorInitArgs:
    sample_rate: int = 24000
    n_fft: int = 1024
    hop_length: int = 256
    n_mels: int = 100
    padding: str = "center"


@dataclass(repr=False, eq=False)
class FeatureExtractor:
    class_path: str = "vocos.feature_extractors.MelSpectrogramFeatures"
    init_args: FeatureExtractorInitArgs = FeatureExtractorInitArgs()


@dataclass(repr=False, eq=False)
class BackboneInitArgs:
    input_channels: int = 100
    dim: int = 512
    intermediate_dim: int = 1536
    num_layers: int = 8


@dataclass(repr=False, eq=False)
class Backbone:
    class_path: str = "vocos.models.VocosBackbone"
    init_args: BackboneInitArgs = BackboneInitArgs()


@dataclass(repr=False, eq=False)
class FourierHeadInitArgs:
    dim: int = 512
    n_fft: int = 1024
    hop_length: int = 256
    padding: str = "center"


@dataclass(repr=False, eq=False)
class FourierHead:
    class_path: str = "vocos.heads.ISTFTHead"
    init_args: FourierHeadInitArgs = FourierHeadInitArgs()


@dataclass(repr=False, eq=False)
class Vocos:
    feature_extractor: FeatureExtractor = FeatureExtractor()
    backbone: Backbone = Backbone()
    head: FourierHead = FourierHead()


@dataclass(repr=False, eq=False)
class LLMProjector:
    llm_dim: int = 4096  # 这个是Qwen-1 不是Qwen2
    use_resampler: bool = False  # 是否使用 Resampler
    query_num: int = 1
    max_size: int = 100


@dataclass(repr=False, eq=False)
class Config:
    decoder: Decoder = Decoder()
    dvae: DVAE = DVAE()
    gpt: GPT = GPT()
    vocos: Vocos = Vocos()
    llm_projector: LLMProjector = LLMProjector()
    pooling: bool = False
    use_speaker_embedding: bool = True
    use_llm_hidden_state: bool = False
    spk_emb_token_id: int = 21143
    num_spk_embs: int = 1
    streaming: bool = False
    streaming_text_chunk_min: int = 3
    streaming_text_chunk_max: int = 7
    streaming_text_reserved_len: int = 150
    tts_chunk_len_s: int = 50
    model_type: str = "default"
    use_text: bool = True
