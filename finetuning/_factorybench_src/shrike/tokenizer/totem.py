"""TOTEM VQ-VAE tokenizer for time series.

Converts continuous time series into discrete token IDs via a pretrained
1D CNN encoder and vector quantizer with a learned 256-entry codebook.

Architecture:
    raw_values → normalize → pad to 4x → Conv1d encoder (4:1 compression)
              → VectorQuantizer (nearest codebook entry) → token IDs [0, 255]

Reference:
    TOTEM: TOkenized Time Series EMbeddings for General Time Series Analysis
    https://arxiv.org/abs/2402.16412
    Pretrained weights: https://drive.google.com/drive/folders/1TSwPHDMAhcpe2AKl4xsVbUUmAvd_Tp-Z
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# VQ-VAE components (from SaberaTalukder/TOTEM, simplified)
# ---------------------------------------------------------------------------
class _Residual(nn.Module):
    def __init__(self, in_channels: int, num_hiddens: int, num_residual_hiddens: int) -> None:
        super().__init__()
        self._block = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels, num_residual_hiddens, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(num_residual_hiddens, num_hiddens, kernel_size=1, stride=1, bias=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self._block(x)


class _ResidualStack(nn.Module):
    def __init__(
        self, in_channels: int, num_hiddens: int,
        num_residual_layers: int, num_residual_hiddens: int) -> None:
        super().__init__()
        self._layers = nn.ModuleList([
            _Residual(in_channels, num_hiddens, num_residual_hiddens)
            for _ in range(num_residual_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self._layers:
            x = layer(x)
        return F.relu(x)


class _TOTEMEncoder(nn.Module):
    """TOTEM 1D CNN encoder with 4:1 compression.

    Architecture:
        (batch, 1, seq_len) → Conv1d(1→32, k=4, s=2) → ReLU
                             → Conv1d(32→64, k=4, s=2) → Conv1d(64→64, k=3)
                             → ResidualStack → Conv1d(64→64, k=1)
                             → (batch, 64, seq_len/4)
    """

    def __init__(
        self,
        num_hiddens: int = 64,
        num_residual_layers: int = 2,
        num_residual_hiddens: int = 128,
        embedding_dim: int = 64) -> None:
        super().__init__()
        self._conv_1 = nn.Conv1d(1, num_hiddens // 2, kernel_size=4, stride=2, padding=1)
        self._conv_2 = nn.Conv1d(num_hiddens // 2, num_hiddens, kernel_size=4, stride=2, padding=1)
        self._conv_3 = nn.Conv1d(num_hiddens, num_hiddens, kernel_size=3, stride=1, padding=1)
        self._residual_stack = _ResidualStack(
            num_hiddens, num_hiddens, num_residual_layers, num_residual_hiddens)
        self._pre_vq_conv = nn.Conv1d(num_hiddens, embedding_dim, kernel_size=1, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self._conv_1(x))
        x = F.relu(self._conv_2(x))
        x = self._conv_3(x)
        x = self._residual_stack(x)
        return self._pre_vq_conv(x)  # (batch, embedding_dim, seq_len/4)


class _VectorQuantizer(nn.Module):
    """Vector quantizer, snaps encoder outputs to nearest codebook entry."""

    def __init__(self, num_embeddings: int = 256, embedding_dim: int = 64) -> None:
        super().__init__()
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._embedding = nn.Embedding(num_embeddings, embedding_dim)
        self._embedding.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: (batch, embedding_dim, seq_len/4)
        inputs = inputs.permute(0, 2, 1).contiguous()  # (batch, seq_len/4, embedding_dim)
        flat_input = inputs.view(-1, self._embedding_dim)

        # L2 distance to all codebook entries
        distances = (
            torch.sum(flat_input ** 2, dim=1, keepdim=True)
            + torch.sum(self._embedding.weight ** 2, dim=1)
            - 2 * torch.matmul(flat_input, self._embedding.weight.t())
        )
        encoding_indices = torch.argmin(distances, dim=1)
        return encoding_indices.view(inputs.shape[0], inputs.shape[1])  # (batch, seq_len/4)

    @property
    def codebook(self) -> torch.Tensor:
        """Return the codebook weight matrix (num_embeddings, embedding_dim)."""
        return self._embedding.weight.data

    @property
    def codebook_dim(self) -> int:
        return self._embedding_dim

    @property
    def codebook_size(self) -> int:
        return self._num_embeddings


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
class _TOTEMDecoder(nn.Module):
    """TOTEM 1D CNN decoder, mirrors the encoder with transposed convolutions.

    Architecture:
        (batch, embedding_dim, seq_len/4) → Conv1d(64→64, k=3) → ResidualStack
        → ConvTranspose1d(64→32, k=4, s=2) → ReLU
        → ConvTranspose1d(32→1, k=4, s=2)
        → (batch, 1, seq_len)
    """

    def __init__(
        self,
        num_hiddens: int = 64,
        num_residual_layers: int = 2,
        num_residual_hiddens: int = 128,
        embedding_dim: int = 64) -> None:
        super().__init__()
        self._conv_1 = nn.Conv1d(embedding_dim, num_hiddens, kernel_size=3, stride=1, padding=1)
        self._residual_stack = _ResidualStack(
            num_hiddens, num_hiddens, num_residual_layers, num_residual_hiddens)
        self._conv_trans_1 = nn.ConvTranspose1d(
            num_hiddens, num_hiddens // 2, kernel_size=4, stride=2, padding=1)
        self._conv_trans_2 = nn.ConvTranspose1d(
            num_hiddens // 2, 1, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._conv_1(x)
        x = self._residual_stack(x)
        x = F.relu(self._conv_trans_1(x))
        return self._conv_trans_2(x)  # (batch, 1, seq_len)


class TOTEMTokenizer(nn.Module):
    """Complete TOTEM tokenizer: 1D CNN encoder + vector quantizer + decoder.

    The encoder and quantizer handle tokenization (values → codes).
    The decoder handles reconstruction (codes → values) for Chameleon-style
    forecasting where the LLM generates code tokens that must be converted
    back to numerical time series.

    Frozen after loading pretrained weights. Runs on CPU to save GPU memory.

    Usage:
        tokenizer = TOTEMTokenizer.from_pretrained("checkpoints/totem_generalist.pt")
        token_ids = tokenizer.tokenize(raw_time_series)  # → [0, 255] integers
        reconstructed = tokenizer.decode(token_ids)       # → float values
    """

    COMPRESSION_FACTOR = 4

    def __init__(
        self,
        num_hiddens: int = 64,
        num_residual_layers: int = 2,
        num_residual_hiddens: int = 128,
        embedding_dim: int = 64,
        num_embeddings: int = 256) -> None:
        super().__init__()
        self.encoder = _TOTEMEncoder(
            num_hiddens, num_residual_layers, num_residual_hiddens, embedding_dim)
        self.quantizer = _VectorQuantizer(num_embeddings, embedding_dim)
        self.decoder = _TOTEMDecoder(
            num_hiddens, num_residual_layers, num_residual_hiddens, embedding_dim)

    def tokenize(self, ts: torch.Tensor) -> torch.Tensor:
        """Convert raw time series to discrete token IDs.

        Args:
            ts: Raw time series values, shape (batch, seq_len) or (seq_len).

        Returns:
            Integer token IDs in [0, num_embeddings), shape (batch, seq_len // 4).
        """
        if ts.dim() == 1:
            ts = ts.unsqueeze(0)

        # Pad to multiple of compression factor
        seq_len = ts.shape[-1]
        pad_len = (self.COMPRESSION_FACTOR - seq_len % self.COMPRESSION_FACTOR) % self.COMPRESSION_FACTOR
        if pad_len > 0:
            ts = F.pad(ts, (0, pad_len))

        x = ts.unsqueeze(1).float()      # (batch, 1, seq_len)
        z = self.encoder(x)               # (batch, embed_dim, seq_len/4)
        return self.quantizer(z)           # (batch, seq_len/4)

    def decode(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Decode discrete code IDs back to continuous time series values.

        This is the reverse of ``tokenize``: looks up codebook vectors for
        each code ID, then runs the CNN decoder to reconstruct the signal.
        Used for Chameleon-style forecasting where the LLM generates
        ``<ts_*>`` tokens and we need numerical predictions.

        Args:
            token_ids: Integer code IDs, shape (batch, n_codes) or (n_codes).

        Returns:
            Reconstructed values, shape (batch, n_codes * 4).
            Values are in normalized space, caller must de-normalize
            using the original signal's mean and std.
        """
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)

        # Look up codebook vectors: (batch, n_codes) → (batch, n_codes, embed_dim)
        quantized = self.quantizer._embedding(token_ids.long())
        # Decoder expects (batch, embed_dim, n_codes)
        quantized = quantized.permute(0, 2, 1).contiguous()
        # Decode: (batch, 1, n_codes * 4)
        reconstructed = self.decoder(quantized)
        return reconstructed.squeeze(1)  # (batch, n_codes * 4)

    @property
    def codebook(self) -> torch.Tensor:
        """The learned codebook: (num_embeddings, embedding_dim)."""
        return self.quantizer.codebook

    @property
    def codebook_dim(self) -> int:
        """Dimension of each codebook entry."""
        return self.quantizer.codebook_dim

    @property
    def codebook_size(self) -> int:
        """Number of codebook entries."""
        return self.quantizer.codebook_size

    @classmethod
    def from_pretrained(cls, checkpoint_path: str, device: str = "cpu") -> TOTEMTokenizer:
        """Load from a pretrained TOTEM checkpoint.

        Args:
            checkpoint_path: Path to the .pt checkpoint file.
            device: Device to load onto (default: CPU to save GPU memory).

        Returns:
            Frozen TOTEMTokenizer in eval mode.
        """
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

        config = ckpt.get("config", {})
        model = cls(
            num_hiddens=config.get("block_hidden_size", config.get("hid", 64)),
            num_residual_layers=config.get("num_residual_layers", 2),
            num_residual_hiddens=config.get("res_hidden_size", config.get("res_hid", 128)),
            embedding_dim=config.get("embedding_dim", 64),
            num_embeddings=config.get("num_embeddings", 256))

        # Load weights, handle different checkpoint formats
        state_dict = ckpt.get("model_state_dict", ckpt.get("model_state", ckpt.get("state_dict", ckpt)))
        if isinstance(state_dict, dict) and not any(k.startswith("encoder") for k in state_dict):
            state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}

        # Map checkpoint keys to our module keys.
        # Handles 3 formats:
        #   1. Our format: encoder._conv_1.weight (direct match)
        #   2. TOTEM original: vq._embedding.weight -> quantizer._embedding.weight
        #   3. Sequential format: encoder.0.weight -> encoder._conv_1.weight
        sequential_map = {
            "encoder.0": "encoder._conv_1",
            "encoder.2": "encoder._conv_2",
            "encoder.3": "encoder._conv_3",
            "encoder.4.block": "encoder._residual_stack._layers.0._block",
            "encoder.5.block": "encoder._residual_stack._layers.1._block",
            "encoder.6": "encoder._pre_vq_conv",
            "decoder.0": "decoder._conv_1",
            "decoder.1.block": "decoder._residual_stack._layers.0._block",
            "decoder.2.block": "decoder._residual_stack._layers.1._block",
            "decoder.3": "decoder._conv_trans_1",
            "decoder.5": "decoder._conv_trans_2",
        }

        mapped = {}
        for k, v in state_dict.items():
            if k in ("vq.cluster_size", "vq.embed_avg"):
                continue  # EMA buffers, skip
            new_k = k
            # VQ key mapping
            if k == "vq.embedding":
                new_k = "quantizer._embedding.weight"
            elif k.startswith("vq."):
                new_k = "quantizer." + k[len("vq."):]
            else:
                # Sequential format mapping
                for old_prefix, new_prefix in sequential_map.items():
                    if k.startswith(old_prefix):
                        new_k = k.replace(old_prefix, new_prefix, 1)
                        break
            mapped[new_k] = v

        # Load matching keys
        model_keys = set(model.state_dict().keys())
        filtered = {k: v for k, v in mapped.items() if k in model_keys}

        if filtered:
            model.load_state_dict(filtered, strict=False)
        else:
            import warnings
            warnings.warn(
                f"No matching weights found in {checkpoint_path}. "
                f"Checkpoint keys: {list(state_dict.keys())[:5]}",
                stacklevel=2)

        model.eval()
        model.requires_grad_(False)
        return model


def ts_bins_to_text(bins: list[int], prefix: str = "<ts_") -> str:
    """Convert token ID list to text representation for prompt embedding.

    Example: [0, 128, 255] → "<ts_0> <ts_128> <ts_255>"
    """
    return " ".join(f"{prefix}{b}>" for b in bins)
