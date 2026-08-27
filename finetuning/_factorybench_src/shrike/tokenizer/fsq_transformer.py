"""FSQ Transformer Tokenizer, Transformer encoder + FSQ quantization.

Inspired by Archetype: a 1D Transformer autoencoder that compresses raw time series
into discrete tokens using FSQ. Each token captures semantically meaningful patterns
because the Transformer gives every position global context before quantization.

Key difference from CNN-based FSQ:
- CNN FSQ: each code sees ~16 raw points (local) → random codes
- Transformer FSQ: each code sees the ENTIRE signal (global) → structured codes

Architecture:
    Raw signal → Patch embedding → Transformer encoder → FSQ → Transformer decoder → Reconstruction

Usage:
    from shrike.tokenizer.fsq_transformer import FSQTransformerTokenizer

    tok = FSQTransformerTokenizer(config)
    codes = tok.tokenize(signal)     # (batch, n_patches) integer codes
    recon = tok.decode_codes(codes)  # (batch, seq_len) reconstructed signal
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class FSQTransformerConfig:
    """Configuration for Transformer-based FSQ tokenizer."""
    # FSQ
    levels: List[int] = field(default_factory=lambda: [7, 7, 7, 7])  # 2401 codes

    # Patching
    patch_size: int = 4          # raw points per patch
    input_length: int = 256      # expected input length (can handle variable)

    # Transformer
    d_model: int = 128           # embedding dimension
    n_heads: int = 4             # attention heads
    n_encoder_layers: int = 4    # encoder depth
    n_decoder_layers: int = 4    # decoder depth
    d_ff: int = 512              # feedforward hidden dim
    dropout: float = 0.1

    # Training
    temporal_weight: float = 0.1  # weight for temporal smoothness loss

    @property
    def codebook_size(self) -> int:
        result = 1
        for l in self.levels:
            result *= l
        return result

    @property
    def fsq_dim(self) -> int:
        return len(self.levels)

    @property
    def n_patches(self) -> int:
        return self.input_length // self.patch_size


class FSQLayer(nn.Module):
    """Finite Scalar Quantization, same as before."""

    def __init__(self, levels: List[int]):
        super().__init__()
        self.levels = levels
        self.dim = len(levels)
        self.register_buffer("_levels", torch.tensor(levels, dtype=torch.float32))
        self.codebook_size = 1
        for l in levels:
            self.codebook_size *= l

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = torch.tanh(z)
        half_width = self._levels / 2
        scaled = z * (half_width - 0.5)
        quantized_raw = torch.round(scaled)
        quantized = scaled + (quantized_raw - scaled).detach()
        quantized_norm = quantized / (half_width - 0.5)
        indices = (quantized_raw + half_width.unsqueeze(0).unsqueeze(0)).long()
        for i, l in enumerate(self.levels):
            indices[..., i] = indices[..., i].clamp(0, l - 1)
        return quantized_norm, indices

    def codes_to_flat(self, indices: torch.Tensor) -> torch.Tensor:
        flat = torch.zeros(indices.shape[:-1], device=indices.device, dtype=torch.long)
        mult = 1
        for i in reversed(range(len(self.levels))):
            flat += indices[..., i] * mult
            mult *= self.levels[i]
        return flat

    def flat_to_codes(self, flat: torch.Tensor) -> torch.Tensor:
        indices = []
        remainder = flat
        for i in reversed(range(len(self.levels))):
            indices.append(remainder % self.levels[i])
            remainder = remainder // self.levels[i]
        indices.reverse()
        return torch.stack(indices, dim=-1)


class PatchEmbedding(nn.Module):
    """Convert raw signal into patch embeddings."""

    def __init__(self, patch_size: int, d_model: int):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(patch_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len) raw signal
        Returns:
            patches: (batch, n_patches, d_model)
        """
        b, l = x.shape
        n = l // self.patch_size
        # Reshape into patches
        x = x[::n * self.patch_size].reshape(b, n, self.patch_size)
        return self.proj(x)


class PatchReconstruct(nn.Module):
    """Convert patch embeddings back to raw signal."""

    def __init__(self, patch_size: int, d_model: int):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(d_model, patch_size)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patches: (batch, n_patches, d_model)
        Returns:
            x: (batch, seq_len) reconstructed signal
        """
        b, n, _ = patches.shape
        x = self.proj(patches)  # (b, n, patch_size)
        return x.reshape(b, n * self.patch_size)


class FSQTransformerTokenizer(nn.Module):
    """Transformer-based FSQ time series tokenizer.

    The Transformer encoder gives each position GLOBAL context before
    FSQ quantization, producing semantically meaningful discrete codes.
    """

    def __init__(self, config: FSQTransformerConfig | None = None):
        super().__init__()
        if config is None:
            config = FSQTransformerConfig()
        self.config = config

        # Patch embedding
        self.patch_embed = PatchEmbedding(config.patch_size, config.d_model)

        # Positional encoding (learnable)
        max_patches = 1024  # support up to 4096-point signals at patch_size=4
        self.pos_embed = nn.Parameter(torch.randn(1, max_patches, config.d_model) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.n_encoder_layers)

        # Project to FSQ dimension
        self.to_fsq = nn.Linear(config.d_model, config.fsq_dim)

        # FSQ quantization
        self.fsq = FSQLayer(config.levels)

        # Project back from FSQ to d_model
        self.from_fsq = nn.Linear(config.fsq_dim, config.d_model)

        # Transformer decoder
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True)
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=config.n_decoder_layers)

        # Reconstruct patches
        self.patch_reconstruct = PatchReconstruct(config.patch_size, config.d_model)

    @property
    def codebook_size(self) -> int:
        return self.fsq.codebook_size

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode raw signal to FSQ codes.

        Args:
            x: (batch, seq_len) raw signal

        Returns:
            quantized: (batch, n_patches, fsq_dim) quantized vectors
            indices: (batch, n_patches, fsq_dim) per-dim indices
            z_pre: (batch, n_patches, fsq_dim) pre-quantization latents (for temporal loss)
        """
        # Patch embed + positional encoding
        h = self.patch_embed(x)  # (b, n_patches, d_model)
        n = h.shape[1]
        h = h + self.pos_embed[::n:]

        # Transformer encoder, global context
        h = self.encoder(h)  # (b, n_patches, d_model)

        # Project to FSQ dim
        z = self.to_fsq(h)  # (b, n_patches, fsq_dim)

        # FSQ quantize
        quantized, indices = self.fsq(z)

        return quantized, indices, z

    def decode(self, quantized: torch.Tensor) -> torch.Tensor:
        """Decode quantized vectors back to signal.

        Args:
            quantized: (batch, n_patches, fsq_dim)

        Returns:
            recon: (batch, seq_len) reconstructed signal
        """
        # Project back to d_model
        h = self.from_fsq(quantized)  # (b, n_patches, d_model)

        # Add positional encoding
        n = h.shape[1]
        h = h + self.pos_embed[::n:]

        # Transformer decoder
        h = self.decoder(h)  # (b, n_patches, d_model)

        # Reconstruct signal
        return self.patch_reconstruct(h)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass.

        Returns:
            recon: (batch, seq_len) reconstructed signal
            indices: (batch, n_patches, fsq_dim)
            z_pre: (batch, n_patches, fsq_dim) for temporal loss
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        quantized, indices, z_pre = self.encode(x)
        recon = self.decode(quantized)
        return recon, indices, z_pre

    def tokenize(self, x: torch.Tensor) -> torch.Tensor:
        """Encode signal to flat code IDs.

        Args:
            x: (batch, seq_len) or (seq_len) raw signal

        Returns:
            codes: (batch, n_patches) flat integer codes
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        _, indices, _ = self.encode(x)
        return self.fsq.codes_to_flat(indices)

    def decode_codes(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode flat code IDs back to signal."""
        indices = self.fsq.flat_to_codes(codes)
        half_width = self.fsq._levels / 2
        quantized_raw = indices.float() - half_width.unsqueeze(0).unsqueeze(0)
        quantized_norm = quantized_raw / (half_width.unsqueeze(0).unsqueeze(0) - 0.5)
        return self.decode(quantized_norm)

    def temporal_smoothness_loss(self, z_pre: torch.Tensor) -> torch.Tensor:
        """Penalize rapid changes in pre-quantization latents.

        Encourages adjacent positions to have similar latent vectors,
        which means they'll quantize to the same or nearby codes.

        Args:
            z_pre: (batch, n_patches, fsq_dim) pre-quantization latents

        Returns:
            Scalar loss
        """
        # L1 difference between adjacent positions
        diff = z_pre[:, 1::] - z_pre[::-1:]
        return diff.abs().mean()

    def trainable_summary(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}

    @classmethod
    def from_pretrained(cls, checkpoint_path: str, device: str = "cpu") -> "FSQTransformerTokenizer":
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        cfg_dict = ckpt.get("config", {})

        config = FSQTransformerConfig(
            levels=cfg_dict.get("levels", [7, 7, 7, 7]),
            patch_size=cfg_dict.get("patch_size", 4),
            input_length=cfg_dict.get("input_length", 256),
            d_model=cfg_dict.get("d_model", 128),
            n_heads=cfg_dict.get("n_heads", 4),
            n_encoder_layers=cfg_dict.get("n_encoder_layers", 4),
            n_decoder_layers=cfg_dict.get("n_decoder_layers", 4),
            d_ff=cfg_dict.get("d_ff", 512),
            dropout=cfg_dict.get("dropout", 0.0),  # no dropout at inference
            temporal_weight=cfg_dict.get("temporal_weight", 0.1))

        model = cls(config).to(device)
        model.load_state_dict(ckpt["model_state"], strict=False)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        print(f"FSQ Transformer loaded: {config.codebook_size} codes, "
              f"levels={config.levels}, patch_size={config.patch_size}, "
              f"d_model={config.d_model}, layers={config.n_encoder_layers}+{config.n_decoder_layers}")
        return model
