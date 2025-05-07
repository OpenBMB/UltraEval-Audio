import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Union, Optional
import numpy as np
from torch.nn.utils import weight_norm
import math
from einops import rearrange
from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils.parametrize import remove_parametrizations


def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


class CausalConv1d(nn.Conv1d):
    def __init__(self, *args, padding: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.__padding = padding

    def forward(self, x):
        x_pad = F.pad(x, (self.__padding * 2, 0))
        return super().forward(x_pad)


class CausalTransposeConv1d(nn.ConvTranspose1d):
    def __init__(self, *args, padding: int = 0, output_padding: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.__padding = padding
        self.__output_padding = output_padding

    def forward(self, x):
        return super().forward(x)[..., : -(self.__padding * 2 - self.__output_padding)]


def WNCausalConv1d(*args, **kwargs):
    return weight_norm(CausalConv1d(*args, **kwargs))


def WNCausalTransposeConv1d(*args, **kwargs):
    return weight_norm(CausalTransposeConv1d(*args, **kwargs))


# Scripting this brings model speed up 1.4x
@torch.jit.script
def snake(x, alpha):
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    x = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
    x = x.reshape(shape)
    return x


class Snake1d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x):
        return snake(x, self.alpha)


class LocalMHA(nn.Module):
    def __init__(self, dim=1024, window_size=32, dim_head=64):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.heads = dim // dim_head
        self.window_size = window_size
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.rel_pos = SinusoidalEmbeddings(dim_head)
        self.to_out = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        B, C, T = x.shape
        residual = x
        x = self.norm(x.transpose(1, 2))

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)

        # (b, n_h, T, d_h)
        q = q.view(B, T, self.heads, -1).permute(0, 2, 1, 3)
        k = k.view(B, T, self.heads, -1).permute(0, 2, 1, 3)
        v = v.view(B, T, self.heads, -1).permute(0, 2, 1, 3)

        q_pos = torch.arange(T, device=x.device)[:, None]
        k_pos = torch.arange(T, device=x.device)[None, :]
        attn_mask = (q_pos >= k_pos) & (q_pos - k_pos < self.window_size)

        pos_emb = self.rel_pos(k)
        q, k = apply_rotary_pos_emb(q, k, pos_emb)

        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask
        )
        out = out.view(B, self.heads, T, -1).permute(0, 2, 1, 3).reshape(B, T, -1)
        out = self.to_out(out)
        return out.transpose(1, 2) + residual


class SinusoidalEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)
        self.register_buffer("scale", scale, persistent=False)

    def forward(self, x):
        seq_len = x.shape[-2]
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum("i , j -> i j", t, self.inv_freq)
        freqs = torch.cat((freqs, freqs), dim=-1)
        return freqs


def rotate_half(x):
    x = x.reshape(*x.shape[:-1], 2, -1)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, freqs):
    q_len = q.shape[-2]
    q_freqs = freqs[..., -q_len:, :]
    q = (q * q_freqs.cos()) + (rotate_half(q) * q_freqs.sin())
    k = (k * freqs.cos()) + (rotate_half(k) * freqs.sin())
    return q, k


@torch.jit.script
def rmsnorm(x, weight, eps: float):
    orig_type = x.dtype

    x = x.float()
    x_norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    x_norm = x_norm.to(orig_type)

    return x_norm * weight


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        """
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        """
        Forward pass through the RMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.

        """
        return rmsnorm(x, self.weight, self.eps)


class TransformerAttentionBlock(nn.Module):
    def __init__(self, dim=1024, window_size=32, dim_head=64, rmsnorm: bool = False):
        super().__init__()
        self.norm = nn.LayerNorm(dim) if not rmsnorm else RMSNorm(dim)
        self.heads = dim // dim_head
        self.window_size = window_size
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.rel_pos = SinusoidalEmbeddings(dim_head)
        self.to_out = nn.Linear(dim, dim, bias=False)

    def forward(self, x, attn_mask=None):
        B, T, C = x.shape
        residual = x
        x = self.norm(x)

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)

        # (b, n_h, T, d_h)
        q = q.view(B, T, self.heads, -1).permute(0, 2, 1, 3)
        k = k.view(B, T, self.heads, -1).permute(0, 2, 1, 3)
        v = v.view(B, T, self.heads, -1).permute(0, 2, 1, 3)

        if attn_mask is None:
            q_pos = torch.arange(T, device=x.device)[:, None]
            k_pos = torch.arange(T, device=x.device)[None, :]
            attn_mask = (q_pos >= k_pos) & (q_pos - k_pos < self.window_size)

        pos_emb = self.rel_pos(k)
        q, k = apply_rotary_pos_emb(q, k, pos_emb)

        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask
        )
        out = out.view(B, self.heads, T, -1).permute(0, 2, 1, 3).reshape(B, T, -1)
        out = self.to_out(out)
        return out + residual


class TransformerFeedforwardBlock(nn.Module):
    def __init__(self, dim=1024, dim_ff=2560, rmsnorm: bool = False):
        super().__init__()
        self.norm = nn.LayerNorm(dim) if not rmsnorm else RMSNorm(dim)
        self.up_proj = nn.Linear(dim, dim_ff, bias=False)
        self.act = nn.ReLU()
        self.gate_proj = nn.Linear(dim, dim_ff, bias=False)

        self.down_proj = nn.Linear(dim_ff, dim, bias=False)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.down_proj(self.up_proj(x) * self.act(self.gate_proj(x)))
        return x + residual


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim_model: int,
        window_size: int,
        dim_head: int,
        dim_ff: int,
        transpose: bool = False,
        rmsnorm: bool = False,
    ):
        super().__init__()
        self.transpose = transpose
        self.attn = TransformerAttentionBlock(dim_model, window_size, dim_head, rmsnorm)
        self.ff = TransformerFeedforwardBlock(dim_model, dim_ff, rmsnorm)

    def forward(self, x, attn_mask=None):
        # (B, D, T) -> (B, T, D)
        x = self.attn(x.transpose(1, 2) if self.transpose else x, attn_mask)
        x = self.ff(x)
        return x.transpose(1, 2) if self.transpose else x


class CausalResidualUnit(nn.Module):
    def __init__(
        self, dim: int = 16, dilation: int = 1, kernel: int = 7, groups: int = 1
    ):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.block = nn.Sequential(
            Snake1d(dim),
            WNCausalConv1d(
                dim,
                dim,
                kernel_size=kernel,
                dilation=dilation,
                padding=pad,
                groups=groups,
            ),
            Snake1d(dim),
            WNCausalConv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x):
        y = self.block(x)
        pad = (x.shape[-1] - y.shape[-1]) // 2
        assert pad == 0
        if pad > 0:
            x = x[..., pad:-pad]
        return x + y


class CausalEncoderBlock(nn.Module):
    def __init__(self, output_dim: int = 16, input_dim=None, stride: int = 1, groups=1):
        super().__init__()
        input_dim = input_dim or output_dim // 2
        self.block = nn.Sequential(
            CausalResidualUnit(input_dim, dilation=1, groups=groups),
            CausalResidualUnit(input_dim, dilation=3, groups=groups),
            CausalResidualUnit(input_dim, dilation=9, groups=groups),
            Snake1d(input_dim),
            WNCausalConv1d(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
        )

    def forward(self, x):
        return self.block(x)


class CausalEncoder(nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        strides: list = [2, 4, 8, 8],
        depthwise: bool = False,
        attn_window_size: int = 32,
    ):
        super().__init__()
        # Create first convolution
        self.block = [WNCausalConv1d(1, d_model, kernel_size=7, padding=3)]

        # Create EncoderBlocks that double channels as they downsample by `stride`
        for stride in strides:
            d_model *= 2
            groups = d_model // 2 if depthwise else 1
            self.block += [
                CausalEncoderBlock(output_dim=d_model, stride=stride, groups=groups)
            ]

        if attn_window_size is not None:
            self.block += [LocalMHA(dim=d_model, window_size=attn_window_size)]

        groups = d_model if depthwise else 1

        # Create last convolution
        self.block += [
            WNCausalConv1d(d_model, d_model, kernel_size=7, padding=3, groups=groups),
        ]

        # Wrap black into nn.Sequential
        self.block = nn.Sequential(*self.block)
        self.enc_dim = d_model

    def forward(self, x):
        return self.block(x)


class CausalDecoderBlock(nn.Module):
    def __init__(
        self, input_dim: int = 16, output_dim: int = 8, stride: int = 1, groups=1
    ):
        super().__init__()
        self.block = nn.Sequential(
            Snake1d(input_dim),
            WNCausalTransposeConv1d(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
                output_padding=stride % 2,
            ),
            CausalResidualUnit(output_dim, dilation=1, groups=groups),
            CausalResidualUnit(output_dim, dilation=3, groups=groups),
            CausalResidualUnit(output_dim, dilation=9, groups=groups),
        )

    def forward(self, x):
        return self.block(x)


class CausalDecoder(nn.Module):
    def __init__(
        self,
        input_channel,
        channels,
        rates,
        depthwise: bool = False,
        attn_window_size: int = 32,
        d_out: int = 1,
        transformer={
            "enable": False,
            "position": 0,
            "num_layers": 1,
            "dim_ff": 3840,
            "dim_head": 64,
            "rmsnorm": True,
        },
    ):
        super().__init__()

        # Add first conv layer
        if depthwise:
            layers = [
                WNCausalConv1d(
                    input_channel,
                    input_channel,
                    kernel_size=7,
                    padding=3,
                    groups=input_channel,
                ),
                WNCausalConv1d(input_channel, channels, kernel_size=1),
            ]
        else:
            layers = [WNCausalConv1d(input_channel, channels, kernel_size=7, padding=3)]

        if attn_window_size is not None:
            layers += [LocalMHA(dim=channels, window_size=attn_window_size)]

        # Add upsampling + MRF blocks
        for i, stride in enumerate(rates):
            if transformer["enable"] and transformer["position"] == i:
                # initialize transformer block
                transformer_blocks = [
                    TransformerBlock(
                        dim_model=channels // 2**i,
                        window_size=attn_window_size * 2**i,
                        dim_head=transformer["dim_head"] // 2**i,
                        dim_ff=transformer["dim_ff"] // 2**i,
                        transpose=True,
                        rmsnorm=transformer["rmsnorm"],
                    )
                    for _ in range(transformer["num_layers"])
                ]
                with torch.no_grad():
                    for block in transformer_blocks:
                        for name, param in block.named_parameters():
                            if ("attn.to_out.weight" in name) or (
                                "ff.down_proj.weight" in name
                            ):
                                param.zero_()
                layers += transformer_blocks

            input_dim = channels // 2**i
            output_dim = channels // 2 ** (i + 1)
            groups = output_dim if depthwise else 1
            layers += [CausalDecoderBlock(input_dim, output_dim, stride, groups=groups)]

        # Add final conv layer
        layers += [
            Snake1d(output_dim),
            WNCausalConv1d(output_dim, d_out, kernel_size=7, padding=3),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class VectorQuantize(nn.Module):
    """
    Implementation of VQ similar to Karpathy's repo:
    https://github.com/karpathy/deep-vector-quantization
    Additionally uses following tricks from Improved VQGAN
    (https://arxiv.org/pdf/2110.04627.pdf):
        1. Factorized codes: Perform nearest neighbor lookup in low-dimensional space
            for improved codebook usage
        2. l2-normalized codes: Converts euclidean distance to cosine similarity which
            improves training stability
    """

    def __init__(self, input_dim: int, codebook_size: int, codebook_dim: int):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim

        self.in_proj = WNConv1d(input_dim, codebook_dim, kernel_size=1)
        self.out_proj = WNConv1d(codebook_dim, input_dim, kernel_size=1)
        self.codebook = nn.Embedding(codebook_size, codebook_dim)

    def forward(self, z):
        """Quantized the input tensor using a fixed codebook and returns
        the corresponding codebook vectors

        Parameters
        ----------
        z : Tensor[B x D x T]

        Returns
        -------
        Tensor[B x D x T]
            Quantized continuous representation of input
        Tensor[1]
            Commitment loss to train encoder to predict vectors closer to codebook
            entries
        Tensor[1]
            Codebook loss to update the codebook
        Tensor[B x T]
            Codebook indices (quantized discrete representation of input)
        Tensor[B x D x T]
            Projected latents (continuous representation of input before quantization)
        """

        # Factorized codes (ViT-VQGAN) Project input into low-dimensional space
        z_e = self.in_proj(z)  # z_e : (B x D x T)
        z_q, indices = self.decode_latents(z_e)

        commitment_loss = F.mse_loss(z_e, z_q.detach(), reduction="none").mean([1, 2])
        codebook_loss = F.mse_loss(z_q, z_e.detach(), reduction="none").mean([1, 2])

        z_q = (
            z_e + (z_q - z_e).detach()
        )  # noop in forward pass, straight-through gradient estimator in backward pass

        z_q = self.out_proj(z_q)

        return z_q, commitment_loss, codebook_loss, indices, z_e

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.codebook.weight)

    def decode_code(self, embed_id):
        return self.embed_code(embed_id).transpose(1, 2)

    def decode_latents(self, latents):
        encodings = rearrange(latents, "b d t -> (b t) d")
        codebook = self.codebook.weight  # codebook: (N x D)

        # L2 normalize encodings and codebook (ViT-VQGAN)
        encodings = F.normalize(encodings)
        codebook = F.normalize(codebook)

        # Compute euclidean distance with codebook
        dist = (
            encodings.pow(2).sum(1, keepdim=True)
            - 2 * encodings @ codebook.t()
            + codebook.pow(2).sum(1, keepdim=True).t()
        )
        indices = rearrange((-dist).max(1)[1], "(b t) -> b t", b=latents.size(0))
        z_q = self.decode_code(indices)
        return z_q, indices


class StridedVectorQuantize(VectorQuantize):
    def __init__(
        self, input_dim: int, codebook_size: int, codebook_dim: int, stride: int = 1
    ):
        super().__init__(input_dim, codebook_size, codebook_dim)
        self.stride = stride

    def forward(self, z):
        if self.stride > 1:
            z = F.avg_pool1d(z, self.stride, self.stride)

        z_q, commitment_loss, codebook_loss, indices, z_e = super().forward(z)

        if self.stride > 1:
            z_q = z_q.repeat_interleave(self.stride, dim=-1)
            z_e = z_e.repeat_interleave(self.stride, dim=-1)

        return z_q, commitment_loss, codebook_loss, indices, z_e


class MultiScaleResidualVectorQuantize(nn.Module):
    """
    Introduced in SoundStream: An end2end neural audio codec
    https://arxiv.org/abs/2107.03312
    """

    def __init__(
        self,
        input_dim: int = 512,
        codebook_size: int = 1024,
        codebook_dim: Union[int, list] = 8,
        vq_strides: List[int] = [8, 4, 2, 1],
        quantizer_dropout: float = 0.0,
    ):
        super().__init__()
        self.n_codebooks = len(vq_strides)

        if isinstance(codebook_dim, int):
            codebook_dim = [codebook_dim for _ in range(self.n_codebooks)]

        self.codebook_dim = codebook_dim
        self.codebook_size = codebook_size

        self.quantizers = nn.ModuleList(
            [
                StridedVectorQuantize(
                    input_dim, codebook_size, codebook_dim[i], stride=vq_strides[i]
                )
                for i in range(self.n_codebooks)
            ]
        )
        self.quantizer_dropout = quantizer_dropout

    def forward(self, z, n_quantizers: int = None, layers: Optional[List[int]] = None):
        """Quantized the input tensor using a fixed set of `n` codebooks and returns
        the corresponding codebook vectors
        Parameters
        ----------
        z : Tensor[B x D x T]
        n_quantizers : int, optional
            No. of quantizers to use
            (n_quantizers < self.n_codebooks ex: for quantizer dropout)
            Note: if `self.quantizer_dropout` is True, this argument is ignored
                when in training mode, and a random number of quantizers is used.
        Returns
        -------
        dict
            A dictionary with the following keys:

            "z" : Tensor[B x D x T]
                Quantized continuous representation of input
            "codes" : Tensor[B x N x T]
                Codebook indices for each codebook
                (quantized discrete representation of input)
            "latents" : Tensor[B x N*D x T]
                Projected latents (continuous representation of input before quantization)
            "vq/commitment_loss" : Tensor[1]
                Commitment loss to train encoder to predict vectors closer to codebook
                entries
            "vq/codebook_loss" : Tensor[1]
                Codebook loss to update the codebook
        """
        z_q = 0
        residual = z
        commitment_loss = 0
        codebook_loss = 0

        codebook_indices = []
        latents = []
        quantized_out = []

        if n_quantizers is None:
            n_quantizers = self.n_codebooks
        if self.training:
            n_quantizers = torch.ones((z.shape[0],)) * self.n_codebooks + 1
            dropout = torch.randint(1, self.n_codebooks + 1, (z.shape[0],))
            n_dropout = int(z.shape[0] * self.quantizer_dropout)
            n_quantizers[:n_dropout] = dropout[:n_dropout]
            n_quantizers = n_quantizers.to(z.device)
        for i, quantizer in enumerate(self.quantizers):
            if self.training is False and i >= n_quantizers:
                break

            z_q_i, commitment_loss_i, codebook_loss_i, indices_i, z_e_i = quantizer(
                residual
            )

            if layers is not None and i in layers:
                quantized_out.append(z_q_i)

            # Create mask to apply quantizer dropout
            mask = (
                torch.full((z.shape[0],), fill_value=i, device=z.device) < n_quantizers
            )
            z_q = z_q + z_q_i * mask[:, None, None]
            residual = residual - z_q_i

            # Sum losses
            commitment_loss += (commitment_loss_i * mask).mean()
            codebook_loss += (codebook_loss_i * mask).mean()

            codebook_indices.append(indices_i)
            latents.append(z_e_i)

        codes = codebook_indices  # torch.stack(codebook_indices, dim=1)
        latents = torch.cat(latents, dim=1)

        return z_q, codes, latents, quantized_out, commitment_loss, codebook_loss

    def from_codes(self, codes: torch.Tensor, n_codebooks=None):
        """Given the quantized codes, reconstruct the continuous representation
        Parameters
        ----------
        codes : Tensor[B x N x T]
            Quantized discrete representation of input
        Returns
        -------
        Tensor[B x D x T]
            Quantized continuous representation of input
        """
        z_q = 0.0

        if n_codebooks is None:
            n_codebooks = self.n_codebooks

        vq_strides = [self.quantizers[i].stride for i in range(n_codebooks)]
        rev_vq_strides = [vq_strides[0] // s for s in vq_strides]
        sum_stride = sum(rev_vq_strides)

        codes = codes.view(-1, sum_stride)

        stride_st = 0
        for i in range(n_codebooks):
            this_stride = rev_vq_strides[i]
            z_p_i = self.quantizers[i].decode_code(
                codes[:, stride_st : stride_st + this_stride].reshape(1, -1)
            )
            stride_st += this_stride

            if vq_strides[i] > 1:
                z_p_i = z_p_i.repeat_interleave(vq_strides[i], dim=-1)

            z_q_i = self.quantizers[i].out_proj(z_p_i)
            z_q = z_q + z_q_i
        return z_q


class SSNAC(nn.Module):
    def __init__(
        self,
        encoder_dim: int = 64,
        encoder_rates: List[int] = [2, 4, 8, 8],
        latent_dim: int = None,
        decoder_dim: int = 1536,
        decoder_rates: List[int] = [8, 8, 4, 2],
        depthwise: bool = True,
        vq_strides: List[int] = [8, 4, 2, 1],
        attn_window_size: int = 32,
        codebook_size: int = 1024,
        codebook_dim: Union[int, list] = 8,
        quantizer_dropout: bool = False,
        sample_rate: int = 44100,
        feature_out: int = 4 * 768,
        transformer={
            "enable": False,
            "position": 0,
            "num_layers": 1,
            "dim_ff": 3840,
            "dim_head": 64,
            "rmsnorm": True,
        },
    ):
        super().__init__()

        self.encoder_dim = encoder_dim
        self.encoder_rates = encoder_rates
        self.decoder_dim = decoder_dim
        self.decoder_rates = decoder_rates
        self.depthwise = depthwise
        self.vq_strides = vq_strides
        self.attn_window_size = attn_window_size
        self.feature_out = feature_out
        self.transformer = dict(transformer)

        if latent_dim is None:
            latent_dim = encoder_dim * (2 ** len(encoder_rates))

        self.latent_dim = latent_dim

        self.hop_length = np.prod(encoder_rates)
        self.encoder = CausalEncoder(
            encoder_dim,
            encoder_rates,
            depthwise=depthwise,
            attn_window_size=attn_window_size,
        )

        self.n_codebooks = len(vq_strides)
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.quantizer = MultiScaleResidualVectorQuantize(
            input_dim=latent_dim,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            vq_strides=vq_strides,
            quantizer_dropout=quantizer_dropout,
        )

        self.decoder = CausalDecoder(
            latent_dim,
            decoder_dim,
            decoder_rates,
            depthwise=depthwise,
            attn_window_size=attn_window_size,
            transformer=self.transformer,
        )

        self.feature_transform = nn.Linear(latent_dim, self.feature_out)
        self.sample_rate = sample_rate

    def accelerate(self):
        for mod in self.modules():
            if mod.__class__.__name__ in [
                "ParametrizedCausalConv1d",
                "ParametrizedCausalTransposeConv1d",
                "ParametrizedConv1d",
            ]:
                remove_parametrizations(mod, "weight")
        self.encoder = torch.compile(self.encoder)
        self.decoder = self.decoder
        self.quantizer = torch.compile(self.quantizer)
        return self

    def preprocess(self, audio_data, sample_rate):
        if sample_rate is None:
            sample_rate = self.sample_rate
        assert sample_rate == self.sample_rate
        pad_to = self.hop_length * self.vq_strides[0]
        length = audio_data.shape[-1]
        right_pad = math.ceil(length / pad_to) * pad_to - length
        audio_data = nn.functional.pad(audio_data, (0, right_pad))

        return audio_data

    def encode(
        self,
        audio_data: torch.Tensor,
        n_quantizers: int = None,
        layers: Optional[List[int]] = None,
    ):
        """Encode given audio data and return quantized latent codes

        Parameters
        ----------
        audio_data : Tensor[B x 1 x T]
            Audio data to encode
        n_quantizers : int, optional
            Number of quantizers to use, by default None
            If None, all quantizers are used.

        Returns
        -------
        dict
            A dictionary with the following keys:
            "z" : Tensor[B x D x T]
                Quantized continuous representation of input
            "codes" : Tensor[B x N x T]
                Codebook indices for each codebook
                (quantized discrete representation of input)
            "latents" : Tensor[B x N*D x T]
                Projected latents (continuous representation of input before quantization)
            "vq/commitment_loss" : Tensor[1]
                Commitment loss to train encoder to predict vectors closer to codebook
                entries
            "vq/codebook_loss" : Tensor[1]
                Codebook loss to update the codebook
            "length" : int
                Number of samples in input audio
        """
        z = self.encoder(audio_data)
        z, codes, latents, quantized_out, commitment_loss, codebook_loss = (
            self.quantizer(z, n_quantizers, layers=layers)
        )
        return z, codes, latents, quantized_out, commitment_loss, codebook_loss

    def decode(self, z: torch.Tensor):
        """Decode given latent codes and return audio data

        Parameters
        ----------
        z : Tensor[B x D x T]
            Quantized continuous representation of input
        length : int, optional
            Number of samples in output audio, by default None

        Returns
        -------
        dict
            A dictionary with the following keys:
            "audio" : Tensor[B x 1 x length]
                Decoded audio data.
        """
        return self.decoder(z)

    def forward(
        self,
        audio_data: torch.Tensor,
        sample_rate: int = None,
        n_quantizers: int = None,
    ):
        """Model forward pass

        Parameters
        ----------
        audio_data : Tensor[B x 1 x T]
            Audio data to encode
        sample_rate : int, optional
            Sample rate of audio data in Hz, by default None
            If None, defaults to `self.sample_rate`
        n_quantizers : int, optional
            Number of quantizers to use, by default None.
            If None, all quantizers are used.

        Returns
        -------
        dict
            A dictionary with the following keys:
            "z" : Tensor[B x D x T]
                Quantized continuous representation of input
            "codes" : Tensor[B x N x T]
                Codebook indices for each codebook
                (quantized discrete representation of input)
            "latents" : Tensor[B x N*D x T]
                Projected latents (continuous representation of input before quantization)
            "vq/commitment_loss" : Tensor[1]
                Commitment loss to train encoder to predict vectors closer to codebook
                entries
            "vq/codebook_loss" : Tensor[1]
                Codebook loss to update the codebook
            "length" : int
                Number of samples in input audio
            "audio" : Tensor[B x 1 x length]
                Decoded audio data.
        """
        length = audio_data.shape[-1]
        audio_data = self.preprocess(audio_data, sample_rate)
        z, codes, latents, quantized_out, commitment_loss, codebook_loss = self.encode(
            audio_data, n_quantizers, layers=[0]
        )
        quantized_feature = self.feature_transform(quantized_out[0].permute(0, 2, 1))

        x = self.decode(z)
        return {
            "audio": x[..., :length],
            "z": z,
            "codes": codes,
            "latents": latents,
            "quantized_feature": quantized_feature,
            "vq/commitment_loss": commitment_loss,
            "vq/codebook_loss": codebook_loss,
        }


def decode(codes, model):
    codes = codes.transpose(0, 1).contiguous()
    with torch.no_grad():
        latents = model.quantizer.from_codes(torch.tensor(codes).cuda())
        audio_hat = model.decode(latents[..., :])
    return audio_hat


def transform_snac(data: torch.Tensor) -> torch.Tensor:
    cb1 = data[0]
    length = cb1.shape[1]
    cb2 = data[1].view(-1).reshape(length, 2).T
    cb3 = data[2].view(-1).reshape(length, 4).T
    codes = torch.cat([cb1, cb2, cb3], dim=0)
    return codes


def main(codes, model):
    with torch.no_grad():
        latents = model.quantizer.from_codes(torch.tensor(codes).cuda())
        audio_hat = model.decode(latents[..., :])
    import soundfile as sf

    sf.write("test.wav", audio_hat[0, 0].cpu().numpy(), 24000)


if __name__ == "__main__":
    import torchaudio
    import time

    snac_path = "/mnt/data/user/tc_agi/zgy/ssnact_24k_dist20.1000.pt"
    state = torch.load(
        "/mnt/data/user/tc_agi/zgy/ssnact_24k_dist20.1000.pt", map_location="cpu"
    )
    snac_model = SSNAC(**state["metadata"]["kwargs"])
    snac_model.load_state_dict(state["state_dict"])
    snac_model = snac_model.cuda().eval()
    snac_model = snac_model.accelerate()
    for i in range(5):
        s = time.time()
        audio, sr = torchaudio.load(
            "/data/liuxin/workspace/MiniCPM/samples/tts_prompt/liu_prompt_24k.wav"
        )
        audio = audio.unsqueeze(0).cuda()
        audio = snac_model.preprocess(audio, sr)
        z, codes, *_ = snac_model.encode(audio)
        print(time.time() - s)
    codes = transform_snac(codes)
    import pickle as pkl

    pkl.dump(
        codes,
        open(
            "/data/liuxin/workspace/MiniCPM/samples/tts_prompt/liu_prompt_24k_ssnac.pkl",
            "wb",
        ),
    )
    print(codes.shape)
    main(codes.transpose(0, 1).contiguous(), snac_model)
