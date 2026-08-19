"""Wavelet coefficient and signal reconstruction loss."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class WaveletReconstructionLoss(nn.Module):
    def __init__(
        self,
        coeff_weight: float = 0.45,
        recon_weight: float = 0.55,
        masked_weight: float = 3.0,
    ) -> None:
        super().__init__()
        self.coeff_weight = float(coeff_weight)
        self.recon_weight = float(recon_weight)
        self.masked_weight = float(masked_weight)

    def forward(
        self,
        prediction: dict[str, Any],
        target_coefficients: dict[str, Any],
        target_signal: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        band_losses = [
            F.smooth_l1_loss(prediction["approximation"], target_coefficients["approximation"])
        ]
        band_losses.extend(
            F.smooth_l1_loss(predicted, target)
            for predicted, target in zip(prediction["details"], target_coefficients["details"])
        )
        coefficient_loss = torch.stack(band_losses).mean()
        pointwise = F.smooth_l1_loss(prediction["reconstruction"], target_signal, reduction="none")
        if mask is None:
            signal_loss = pointwise.mean()
        else:
            weights = 1.0 + mask.float() * (self.masked_weight - 1.0)
            signal_loss = (pointwise * weights).sum() / weights.sum().clamp_min(1.0)
        total = self.coeff_weight * coefficient_loss + self.recon_weight * signal_loss
        return total, {
            "reconstruction_total": float(total.detach()),
            "reconstruction_coeff": float(coefficient_loss.detach()),
            "reconstruction_signal": float(signal_loss.detach()),
        }
