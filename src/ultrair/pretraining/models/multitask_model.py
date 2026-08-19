"""UltraIR encoder with the three pretraining task heads."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from ultrair.models import UltraIRClassifier

from .heads import ProjectionHead, WaveletReconstructionHead


class UltraIRPretrainingModel(nn.Module):
    def __init__(
        self,
        signal_size: int,
        num_fgroups: int = 17,
        d_model: int = 1024,
        patch_len: int = 16,
        n_heads: int = 16,
        num_global_layers: int = 8,
        dropout: float = 0.1,
        head_dropout: float = 0.3,
        fingerprint_proj_dim: int = 256,
        wavelet: str = "db4",
        wavelet_level: int = 4,
        wavelet_hidden_dim: int | None = None,
        wavelet_bottleneck_dim: int | None = None,
        wavelet_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = UltraIRClassifier(
            num_fgroups=num_fgroups,
            d_model=d_model,
            signal_size=signal_size,
            patch_len=patch_len,
            n_heads=n_heads,
            num_global_layers=num_global_layers,
            dropout=dropout,
            head_dropout=head_dropout,
        )
        self.fingerprint_head = ProjectionHead(d_model, fingerprint_proj_dim, head_dropout)
        self.wavelet_head = WaveletReconstructionHead(
            embedding_dim=d_model,
            signal_size=signal_size,
            wavelet=wavelet,
            level=wavelet_level,
            hidden_dim=wavelet_hidden_dim,
            bottleneck_dim=wavelet_bottleneck_dim,
            dropout=wavelet_dropout,
        )

    def encode(self, signal: torch.Tensor) -> torch.Tensor:
        return self.encoder.forward_features(signal.unsqueeze(1))

    def forward(self, signal: torch.Tensor) -> dict[str, Any]:
        embedding = self.encode(signal)
        return {
            "embedding": embedding,
            "fg_logits": self.encoder.classify_features(embedding),
            "reconstruction": self.wavelet_head(embedding),
            "fingerprint_embedding": self.fingerprint_head(embedding),
        }
