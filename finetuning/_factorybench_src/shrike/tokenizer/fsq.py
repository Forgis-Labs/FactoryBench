"""FSQ (Finite Scalar Quantization) Time Series Tokenizer.

No codebook. No collapse. Just rounding to a finite grid.
Each dimension of the latent is rounded to one of L levels,
giving product(levels) total discrete codes.

Reference: Mentzer et al., "Finite Scalar Quantization: VQ-VAE Made Simple" (ICLR 2024)

Usage:
    from shrike.tokenizer.fsq import FSQTokenizer

    tokenizer = FSQTokenizer.from_pretrained("checkpoints/fsq_4096.pt")
    codes = tokenizer.tokenize(signal)    # (batch, seq_len/4) integer codes
    recon = tokenizer.decode(codes)       # (batch, seq_len) reconstructed signal
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class FSQConfig:
    """Configuration for FSQ tokenizer."""
    levels: List[int] = field(default_factory=lambda: [8, 8, 8, 8])  # 4096 codes
    input_length: int = 256
    compression: int = 4       # 4:1 compression (256 pts -> 64 codes)
    hid: int = 128             # encoder/decoder hidden channels
    n_res: int = 4             # residual blocks
    res_hid: int = 256         # residual block hidden dim

    @property
    def codebook_size(self) -> int:
        result = 1
        for l in self.levels:
            result *= l
        return result

    @property
    def dim(self) -> int:
        return len(self.levels)


class FSQLayer(nn.Module):
    """Finite Scalar Quantization, replaces VQ codebook lookup with rounding."""

    def __init__(self, levels: List[int]):
        super().__init__()
        self.levels = levels
        self.dim = len(levels)
        self.register_buffer("_levels", torch.tensor(levels, dtype=torch.float32))

        self.codebook_size = 1
        for l in levels:
            self.codebook_size *= l

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize latent vectors to finite grid.

        Args:
            z: (batch, seq_len, dim) continuous latent vectors

        Returns:
            quantized: (batch, seq_len, dim) quantized vectors (with STE gradients)
            indices: (batch, seq_len, dim) per-dimension integer indices
        """
        # Bound to [-1, 1] via tanh
        z = torch.tanh(z)

        # Scale to level range and round
        half_width = self._levels / 2
        scaled = z * (half_width - 0.5)
        quantized_raw = torch.round(scaled)

        # Straight-through estimator: forward uses rounded, backward uses continuous
        quantized = scaled + (quantized_raw - scaled).detach()

        # Normalize back to [-1, 1]
        quantized_norm = quantized / (half_width - 0.5)

        # Compute per-dimension indices (0 to L-1)
        indices = (quantized_raw + half_width.unsqueeze(0).unsqueeze(0)).long()
        for i, l in enumerate(self.levels):
            indices[..., i] = indices[..., i].clamp(0, l - 1)

        return quantized_norm, indices

    def codes_to_flat(self, indices: torch.Tensor) -> torch.Tensor:
        """Convert per-dimension indices to flat code IDs.

        Args:
            indices: (batch, seq_len, dim) per-dimension indices

        Returns:
            flat: (batch, seq_len) flat code IDs in [0, codebook_size)
        """
        flat = torch.zeros(indices.shape[:-1], device=indices.device, dtype=torch.long)
        mult = 1
        for i in reversed(range(len(self.levels))):
            flat += indices[..., i] * mult
            mult *= self.levels[i]
        return flat

    def flat_to_codes(self, flat: torch.Tensor) -> torch.Tensor:
        """Convert flat code IDs back to per-dimension indices."""
        indices = []
        remainder = flat
        for i in reversed(range(len(self.levels))):
            indices.append(remainder % self.levels[i])
            remainder = remainder // self.levels[i]
        indices.reverse()
        return torch.stack(indices, dim=-1)


class _Residual(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(dim, hidden, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv1d(hidden, dim, 1, bias=False))

    def forward(self, x):
        return x + self.block(x)


class FSQTokenizer(nn.Module):
    """FSQ-based time series tokenizer.

    Architecture: 1D CNN encoder -> FSQ quantization -> 1D CNN decoder
    Same interface as TOTEMTokenizer for drop-in replacement.
    """

    def __init__(self, config: FSQConfig | None = None):
        super().__init__()
        if config is None:
            config = FSQConfig()
        self.config = config

        # Encoder: (batch, 1, input_length) -> (batch, dim, input_length/4)
        self.encoder = nn.Sequential(
            nn.Conv1d(1, config.hid // 2, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(config.hid // 2, config.hid, 4, stride=2, padding=1),
            nn.Conv1d(config.hid, config.hid, 3, padding=1),
            *[_Residual(config.hid, config.res_hid) for _ in range(config.n_res)],
            nn.Conv1d(config.hid, config.dim, 1))

        # FSQ quantization layer
        self.fsq = FSQLayer(config.levels)

        # Decoder: (batch, dim, input_length/4) -> (batch, 1, input_length)
        self.decoder = nn.Sequential(
            nn.Conv1d(config.dim, config.hid, 3, padding=1),
            *[_Residual(config.hid, config.res_hid) for _ in range(config.n_res)],
            nn.ConvTranspose1d(config.hid, config.hid // 2, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(config.hid // 2, 1, 4, stride=2, padding=1))

    @property
    def codebook_size(self) -> int:
        return self.fsq.codebook_size

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode raw signal to quantized latent + indices.

        Args:
            x: (batch, seq_len) or (batch, 1, seq_len) raw signal

        Returns:
            quantized: (batch, seq_len/4, dim) quantized latent vectors
            indices: (batch, seq_len/4, dim) per-dimension indices
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)
        z = self.encoder(x).permute(0, 2, 1)  # (batch, seq/4, dim)
        return self.fsq(z)

    def decode(self, quantized: torch.Tensor) -> torch.Tensor:
        """Decode quantized latent back to signal.

        Args:
            quantized: (batch, seq_len/4, dim) quantized latent vectors

        Returns:
            recon: (batch, seq_len) reconstructed signal
        """
        return self.decoder(quantized.permute(0, 2, 1)).squeeze(1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Full forward: encode -> quantize -> decode."""
        quantized, indices = self.encode(x)
        recon = self.decode(quantized)
        return recon, indices

    def tokenize(self, x: torch.Tensor) -> torch.Tensor:
        """Encode signal to flat code IDs.

        Args:
            x: (batch, seq_len) or (seq_len) raw signal

        Returns:
            codes: (batch, seq_len/4) flat integer codes in [0, codebook_size)
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        _, indices = self.encode(x)
        return self.fsq.codes_to_flat(indices)

    def decode_codes(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode flat code IDs back to signal.

        Args:
            codes: (batch, seq_len/4) flat integer codes

        Returns:
            recon: (batch, seq_len) reconstructed signal
        """
        indices = self.fsq.flat_to_codes(codes)
        # Convert indices back to quantized values
        half_width = self.fsq._levels / 2
        quantized_raw = indices.float() - half_width.unsqueeze(0).unsqueeze(0)
        quantized_norm = quantized_raw / (half_width.unsqueeze(0).unsqueeze(0) - 0.5)
        return self.decode(quantized_norm)

    @classmethod
    def from_pretrained(cls, checkpoint_path: str, device: str = "cpu") -> "FSQTokenizer":
        """Load a pretrained FSQ tokenizer.

        Args:
            checkpoint_path: Path to .pt checkpoint file.
            device: Target device.

        Returns:
            FSQTokenizer with loaded weights, in eval mode.
        """
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        cfg_dict = ckpt.get("config", {})

        config = FSQConfig(
            levels=cfg_dict.get("levels", [8, 8, 8, 8]),
            input_length=cfg_dict.get("input_length", 256),
            compression=cfg_dict.get("compression", 4),
            hid=cfg_dict.get("hid", 128),
            n_res=cfg_dict.get("n_res", 4),
            res_hid=cfg_dict.get("res_hid", 256))

        model = cls(config).to(device)
        model.load_state_dict(ckpt["model_state"], strict=False)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        print(f"FSQ tokenizer loaded: {config.codebook_size} codes, "
              f"levels={config.levels}, compression={config.compression}:1")
        return model
