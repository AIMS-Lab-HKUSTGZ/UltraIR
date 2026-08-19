"""Fractional contribution estimation for a target mixture component."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn

from ultrair.datasets.npy_dataset import GroupedIndexSampler
from .metrics import regression_statistics


@dataclass
class TargetedFractionalContributionEstimationTask:
    label_file: str = "mixture_ref_weight.npy"
    ref_idx_file: str = "mixture_ref_idx.npy"
    ref_idx_key: str = "ref_idx"
    target_name: str = "target_fractional_contribution"
    positive_threshold: float = 1e-8
    grouped_train_sampling: bool = True
    sampler_seed: int = 42

    @property
    def name(self) -> str:
        return "targeted_fractional_contribution_estimation"

    @property
    def label_filename(self) -> str:
        return self.label_file

    @property
    def num_outputs(self) -> int:
        return 1

    def extra_filenames(self) -> dict[str, str]:
        return {self.ref_idx_key: self.ref_idx_file}

    def class_names(self) -> list[str]:
        return [self.target_name]

    def build_criterion(self) -> nn.Module:
        return nn.MSELoss(reduction="none")

    def prepare_targets(self, targets: torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(targets).float().view(-1, 1)

    def forward_model(self, model: nn.Module, signal: Any) -> torch.Tensor:
        values = signal.get("ir") if isinstance(signal, dict) else signal
        if values.ndim != 3 or values.shape[1] != 2:
            raise ValueError(f"Targeted estimation expects [B,2,L], got {tuple(values.shape)}")
        return model(signal)

    def build_sampler(self, dataset, split: str, batch_size: int, shuffle: bool):
        if split != "train" or not self.grouped_train_sampling:
            return None
        group_ids = dataset.extras.get(self.ref_idx_key)
        return (
            GroupedIndexSampler(group_ids, batch_size, shuffle, self.sampler_seed)
            if group_ids is not None else None
        )

    def compute_loss(self, model_out, targets, criterion, **_kwargs):
        predictions, targets = model_out.view(-1, 1), targets.view(-1, 1)
        mask = targets[:, 0] > self.positive_threshold
        return criterion(predictions[mask], targets[mask]).mean() if mask.any() else predictions.sum() * 0.0

    @torch.no_grad()
    def eval_from_logits_and_targets(
        self, predictions: torch.Tensor, targets: torch.Tensor, sample_indices: Optional[torch.Tensor] = None
    ):
        pred, truth = predictions.cpu().view(-1).numpy(), targets.cpu().view(-1).numpy()
        mask = truth > self.positive_threshold
        stats = regression_statistics(truth[mask], pred[mask]) if mask.any() else {
            "mae": float("nan"), "rmse": float("nan"), "r2": float("nan")
        }
        result = {
            "overall": {**stats, "is_regression": True, "num_samples": int(mask.sum())},
            "per_class": {self.target_name: stats},
            "class_names": self.class_names(),
        }
        if sample_indices is not None:
            result["sample_indices"] = sample_indices.cpu().numpy()[mask].tolist()
        return result
