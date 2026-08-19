"""Reusable single-label classification task for application datasets."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from .metrics import multiclass_statistics


@dataclass
class MulticlassClassificationTask:
    task_name: str
    label_file: str = "labels.npy"
    label_smoothing: float = 0.0
    meta_npy_path: Optional[str] = None
    meta_key: str = "class_names"
    names: Optional[list[str]] = None
    _names_cache: Optional[list[str]] = field(default=None, init=False, repr=False)

    @property
    def name(self) -> str:
        return self.task_name

    @property
    def label_filename(self) -> str:
        return self.label_file

    def class_names(self) -> Optional[list[str]]:
        if self.names is not None:
            return list(self.names)
        if self._names_cache is not None:
            return self._names_cache
        if not self.meta_npy_path:
            return None
        value = np.load(Path(self.meta_npy_path).expanduser(), allow_pickle=True)
        if isinstance(value, np.ndarray) and value.shape == () and value.dtype == object:
            value = value.item()
        if isinstance(value, dict):
            value = value[self.meta_key]
        self._names_cache = [str(item) for item in np.asarray(value).reshape(-1).tolist()]
        return self._names_cache

    @property
    def num_outputs(self) -> Optional[int]:
        names = self.class_names()
        return len(names) if names else None

    def build_criterion(self) -> nn.Module:
        return nn.CrossEntropyLoss(label_smoothing=float(self.label_smoothing))

    def prepare_targets(self, targets: torch.Tensor) -> torch.Tensor:
        targets = torch.as_tensor(targets)
        if targets.ndim == 2 and targets.shape[1] == 1:
            targets = targets[:, 0]
        elif targets.ndim == 2:
            targets = targets.argmax(dim=1)
        return targets.long()

    def forward_model(self, model: nn.Module, signal: torch.Tensor) -> torch.Tensor:
        return model(signal.unsqueeze(1))

    @torch.no_grad()
    def eval_from_logits_and_targets(self, logits: torch.Tensor, targets: torch.Tensor):
        y_true = self.prepare_targets(targets).cpu().numpy()
        y_pred = logits.detach().cpu().argmax(dim=1).numpy()
        overall, per_class, confusion = multiclass_statistics(y_true, y_pred, logits.shape[1])
        overall["is_multiclass"] = True
        names = self.class_names()
        return {
            "overall": overall,
            "per_class": per_class,
            "per_class_named": (
                {names[i]: per_class[i] for i in range(len(names))} if names else {}
            ),
            "confusion_matrix": confusion,
            "class_names": names,
        }
