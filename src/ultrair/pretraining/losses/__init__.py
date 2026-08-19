"""Losses for UltraIR multitask pretraining."""

from .fingerprint import SoftTanimotoContrastiveLoss
from .reconstruction import WaveletReconstructionLoss

__all__ = ["SoftTanimotoContrastiveLoss", "WaveletReconstructionLoss"]
