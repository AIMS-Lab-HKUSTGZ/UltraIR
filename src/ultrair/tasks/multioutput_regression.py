"""Reusable normalized multi-output spectral regression task."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .metrics import normalized_regression_statistics, regression_statistics


@dataclass
class MultioutputRegressionTask:
    task_name: str
    label_file: str = "labels.npy"
    stats_npy_path: Optional[str] = None
    labels_npy_path: Optional[str] = None
    meta_npy_path: Optional[str] = None
    property_names: Optional[list[str]] = None
    stats_mode: str = "global"
    labels_are_normalized: bool = False
    target_normalization: str = "minmax"
    prediction_activation: str = "sigmoid"
    loss: str = "mse"
    loss_beta: float = 0.1
    loss_scale: float = 1.0
    target_indices: Optional[list[int]] = None
    target_weights: Optional[list[float]] = None
    _fold_stats: dict[int, np.ndarray] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.property_names is None:
            self.property_names = self._load_names()
        if self.target_indices is not None:
            self.target_indices = [int(index) for index in self.target_indices]
            if not self.target_indices:
                raise ValueError("target_indices cannot be empty")
            self.property_names = [self.property_names[index] for index in self.target_indices]
        if self.stats_mode == "global":
            self._set_stats(self._load_stats())
        elif self.stats_mode != "per_fold_train":
            raise ValueError("stats_mode must be 'global' or 'per_fold_train'")

    def _load_names(self) -> list[str]:
        if self.meta_npy_path:
            value = np.load(Path(self.meta_npy_path).expanduser(), allow_pickle=True)
            if isinstance(value, np.ndarray) and value.shape == () and value.dtype == object:
                value = value.item()
            if isinstance(value, dict):
                value = value.get("property_names", value.get("class_names"))
            if value is not None:
                if isinstance(value, np.ndarray) and value.ndim == 2:
                    value = value[0]
                return [str(item) for item in np.asarray(value).reshape(-1).tolist()]
        path = self.labels_npy_path or self.stats_npy_path
        if path:
            array = np.load(Path(path).expanduser(), mmap_mode="r")
            width = int(array.shape[1] if array.ndim == 2 else array.shape[0])
            return [f"target_{index}" for index in range(width)]
        raise ValueError("Regression tasks require property_names, meta_npy_path, or a label/statistics file")

    def _load_stats(self) -> np.ndarray:
        normalization = self.target_normalization.lower()
        if normalization in {"minmax", "min_max", "range"} and self.stats_npy_path:
            stats = np.load(Path(self.stats_npy_path).expanduser()).astype(np.float32)
            if self.target_indices is not None:
                stats = stats[self.target_indices]
            return stats
        if self.labels_npy_path:
            labels = np.load(Path(self.labels_npy_path).expanduser()).astype(np.float32)
            labels = self._select_targets(labels)
            if normalization in {"minmax", "min_max", "range"}:
                return np.stack([labels.min(0), labels.max(0)], axis=1)
            if normalization in {"standard", "zscore", "z_score"}:
                return np.stack([labels.mean(0), labels.std(0)], axis=1)
            raise ValueError(f"Unknown target_normalization: {self.target_normalization}")
        if self.labels_are_normalized:
            return np.stack(
                [np.zeros(len(self.property_names)), np.ones(len(self.property_names))], axis=1
            ).astype(np.float32)
        raise ValueError("Global normalization requires stats_npy_path or labels_npy_path")

    def _set_stats(self, stats: np.ndarray) -> None:
        if stats.shape != (len(self.property_names), 2):
            raise ValueError(
                f"Expected normalization statistics [{len(self.property_names)}, 2], got {stats.shape}"
            )
        stats = stats.astype(np.float32, copy=False)
        normalization = self.target_normalization.lower()
        if normalization in {"minmax", "min_max", "range"}:
            self.center = stats[:, 0]
            self.scale = stats[:, 1] - stats[:, 0] + 1e-10
        elif normalization in {"standard", "zscore", "z_score"}:
            self.center = stats[:, 0]
            self.scale = np.maximum(stats[:, 1], 1e-10)
        else:
            raise ValueError(f"Unknown target_normalization: {self.target_normalization}")

    def _select_targets(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        return values if self.target_indices is None else values[..., self.target_indices]

    @property
    def name(self) -> str:
        return self.task_name

    @property
    def label_filename(self) -> str:
        return self.label_file

    @property
    def num_outputs(self) -> int:
        return len(self.property_names)

    def class_names(self) -> list[str]:
        return list(self.property_names)

    def configure_data_context(self, cfg, fold: int, train_dir: str, **kwargs) -> None:
        if self.stats_mode != "per_fold_train":
            return
        if fold not in self._fold_stats:
            labels = np.load(
                Path(kwargs.get("train_label_path", Path(train_dir) / self.label_file))
            ).astype(np.float32)
            labels = self._select_targets(labels)
            if self.target_normalization.lower() in {"standard", "zscore", "z_score"}:
                stats = np.stack([labels.mean(0), labels.std(0)], axis=1)
            else:
                stats = np.stack([labels.min(0), labels.max(0)], axis=1)
            self._fold_stats[fold] = stats
        self._set_stats(self._fold_stats[fold])

    def normalize(self, targets: np.ndarray) -> np.ndarray:
        targets = self._select_targets(targets)
        if self.labels_are_normalized:
            return targets.astype(np.float32, copy=False)
        return (targets - self.center) / self.scale

    def denormalize(self, targets: np.ndarray) -> np.ndarray:
        return np.asarray(targets, dtype=np.float32) * self.scale + self.center

    def build_criterion(self) -> nn.Module:
        mode = self.loss.lower()
        weights = torch.as_tensor(
            self.target_weights or [1.0] * self.num_outputs, dtype=torch.float32
        )
        if weights.shape != (self.num_outputs,):
            raise ValueError(f"target_weights must contain {self.num_outputs} values")
        if mode in {"mse", "l2", "mean_squared_error"}:
            criterion: nn.Module = nn.MSELoss()
        elif mode in {"target_weighted_mse", "weighted_mse"}:
            criterion = TargetWeightedMSELoss(weights)
        elif mode in {"smooth_l1", "smoothl1", "huber"}:
            criterion = nn.SmoothL1Loss(beta=float(self.loss_beta))
        elif mode in {"target_weighted_smooth_l1", "weighted_smooth_l1", "target_weighted_huber"}:
            criterion = TargetWeightedSmoothL1Loss(weights, self.loss_beta)
        else:
            raise ValueError(f"Unknown regression loss: {self.loss}")
        return criterion if self.loss_scale == 1.0 else ScaledLoss(criterion, self.loss_scale)

    def forward_model(self, model: nn.Module, signal: torch.Tensor) -> torch.Tensor:
        predictions = model(signal.unsqueeze(1))
        activation = self.prediction_activation.lower()
        if activation in {"linear", "identity", "none"}:
            return predictions
        if activation == "sigmoid":
            return torch.sigmoid(predictions)
        if activation in {"sigmoid_margin_005", "sigmoid_margin_0.05"}:
            return torch.sigmoid(predictions) * 1.10 - 0.05
        if activation in {"sigmoid_margin_010", "sigmoid_margin_0.10"}:
            return torch.sigmoid(predictions) * 1.20 - 0.10
        raise ValueError(f"Unknown prediction_activation: {self.prediction_activation}")

    @torch.no_grad()
    def eval_from_logits_and_targets(self, predictions: torch.Tensor, targets: torch.Tensor):
        y_pred = self.denormalize(predictions.detach().cpu().numpy())
        y_true = self.denormalize(targets.detach().cpu().numpy())
        per_class = {
            name: regression_statistics(y_true[:, index], y_pred[:, index])
            for index, name in enumerate(self.property_names)
        }
        overall = {
            **regression_statistics(y_true, y_pred),
            **normalized_regression_statistics(y_true, y_pred),
            "is_regression": True,
        }
        return {
            "overall": overall,
            "per_class": per_class,
            "class_names": self.class_names(),
        }


class TargetWeightedMSELoss(nn.Module):
    def __init__(self, weights: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("weights", weights.view(1, -1))

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return ((predictions - targets).square() * self.weights).mean()


class TargetWeightedSmoothL1Loss(nn.Module):
    def __init__(self, weights: torch.Tensor, beta: float) -> None:
        super().__init__()
        self.register_buffer("weights", weights.view(1, -1))
        self.beta = float(beta)

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss = F.smooth_l1_loss(predictions, targets, beta=self.beta, reduction="none")
        return (loss * self.weights).mean()


class ScaledLoss(nn.Module):
    def __init__(self, criterion: nn.Module, scale: float) -> None:
        super().__init__()
        if scale <= 0:
            raise ValueError("loss_scale must be positive")
        self.criterion, self.scale = criterion, float(scale)

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.criterion(predictions, targets) * self.scale
