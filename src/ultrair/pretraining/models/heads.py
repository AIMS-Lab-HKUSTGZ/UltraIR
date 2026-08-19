"""Projection and wavelet reconstruction heads."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWT1DForward, DWT1DInverse


class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, projection_dim: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, projection_dim),
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.network(embedding)


class WaveletReconstructionHead(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        signal_size: int,
        wavelet: str = "db4",
        level: int = 4,
        hidden_dim: int | None = None,
        bottleneck_dim: int | None = None,
        dropout: float = 0.1,
        mode: str = "symmetric",
    ) -> None:
        super().__init__()
        self.signal_size = int(signal_size)
        self.forward_wavelet = DWT1DForward(wave=wavelet, J=int(level), mode=mode)
        self.inverse_wavelet = DWT1DInverse(wave=wavelet, mode=mode)
        hidden_dim = int(hidden_dim or embedding_dim)
        bottleneck_dim = int(bottleneck_dim or max(64, min(hidden_dim, embedding_dim // 4)))
        self.shared = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.bottleneck = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        with torch.no_grad():
            approximation, details = self.forward_wavelet(torch.zeros(1, 1, self.signal_size))
        self.approximation_head = nn.Linear(bottleneck_dim, int(approximation.shape[-1]))
        self.detail_heads = nn.ModuleList(
            nn.Linear(bottleneck_dim, int(detail.shape[-1])) for detail in details
        )
        self.log_scales = nn.Parameter(torch.zeros(1 + len(details), dtype=torch.float32))

    def _fp32(self, tensor: torch.Tensor):
        return torch.autocast(device_type=tensor.device.type, enabled=False)

    def decompose(self, signal: torch.Tensor) -> dict[str, Any]:
        if signal.ndim != 2:
            raise ValueError(f"Expected signal [B, L], got {tuple(signal.shape)}")
        with self._fp32(signal):
            approximation, details = self.forward_wavelet(signal.float().unsqueeze(1))
        return {
            "approximation": approximation.squeeze(1),
            "details": [detail.squeeze(1) for detail in details],
        }

    def reconstruct(self, approximation: torch.Tensor, details: list[torch.Tensor]) -> torch.Tensor:
        with self._fp32(approximation):
            reconstruction = self.inverse_wavelet(
                (approximation.float().unsqueeze(1), [detail.float().unsqueeze(1) for detail in details])
            ).squeeze(1)
        if reconstruction.shape[1] > self.signal_size:
            return reconstruction[:, : self.signal_size]
        if reconstruction.shape[1] < self.signal_size:
            return F.pad(reconstruction, (0, self.signal_size - reconstruction.shape[1]))
        return reconstruction

    def forward(self, embedding: torch.Tensor) -> dict[str, Any]:
        with self._fp32(embedding):
            features = self.bottleneck(self.shared(embedding.float()))
            approximation = self.approximation_head(features) * self.log_scales[0].exp()
            details = [
                head(features) * self.log_scales[index + 1].exp()
                for index, head in enumerate(self.detail_heads)
            ]
            reconstruction = self.reconstruct(approximation, details)
        return {
            "approximation": approximation,
            "details": details,
            "reconstruction": reconstruction,
        }
