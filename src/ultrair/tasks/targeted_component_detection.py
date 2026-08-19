"""Binary detection of a target component in a mixture/reference spectrum pair."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn

from ultrair.datasets.npy_dataset import GroupedIndexSampler
from .metrics import average_precision, binary_statistics, roc_auc


@dataclass
class TargetedComponentDetectionTask:
    label_file: str = "mixture_ref_label.npy"
    ref_idx_file: str = "mixture_ref_idx.npy"
    ref_idx_key: str = "ref_idx"
    threshold: float = 0.5
    grouped_train_sampling: bool = True
    sampler_seed: int = 42

    @property
    def name(self) -> str:
        return "targeted_component_detection"

    @property
    def label_filename(self) -> str:
        return self.label_file

    @property
    def num_outputs(self) -> int:
        return 1

    def extra_filenames(self) -> dict[str, str]:
        return {self.ref_idx_key: self.ref_idx_file}

    def class_names(self) -> list[str]:
        return ["absent", "present"]

    def build_criterion(self) -> nn.Module:
        return nn.BCEWithLogitsLoss()

    def prepare_targets(self, targets: torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(targets).float().view(-1, 1)

    def forward_model(self, model: nn.Module, signal: Any) -> torch.Tensor:
        values = signal.get("ir") if isinstance(signal, dict) else signal
        if values.ndim != 3 or values.shape[1] != 2:
            raise ValueError(f"Targeted detection expects [B,2,L], got {tuple(values.shape)}")
        return model(signal)

    def build_sampler(self, dataset, split: str, batch_size: int, shuffle: bool):
        if split != "train" or not self.grouped_train_sampling:
            return None
        group_ids = dataset.extras.get(self.ref_idx_key)
        return (
            GroupedIndexSampler(group_ids, batch_size, shuffle, self.sampler_seed)
            if group_ids is not None else None
        )

    @torch.no_grad()
    def eval_from_logits_and_targets(
        self, logits: torch.Tensor, targets: torch.Tensor, sample_indices: Optional[torch.Tensor] = None
    ):
        scores = torch.sigmoid(logits.detach()).cpu().view(-1).numpy()
        truth = targets.cpu().view(-1).numpy().astype(np.int32)
        prediction = (scores >= self.threshold).astype(np.int32)
        overall = binary_statistics(truth, prediction)
        negative = binary_statistics(1 - truth, 1 - prediction)
        overall.update({
            "is_binary": True,
            "threshold": self.threshold,
            # The paper reports the unweighted mean of absent/present
            # one-vs-rest F1 scores, rather than only the positive-class F1.
            "macro_f1": float((overall["f1"] + negative["f1"]) / 2.0),
            "roc_auc": roc_auc(truth, scores),
            "average_precision": average_precision(truth, scores),
            "num_samples": len(truth),
        })
        result = {"overall": overall, "class_names": self.class_names()}
        if sample_indices is not None:
            result["sample_indices"] = sample_indices.cpu().tolist()
        return result
