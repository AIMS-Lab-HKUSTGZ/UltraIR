"""Functional-group multilabel prediction."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn

from .metrics import binary_statistics


FUNCTIONAL_GROUP_NAMES = (
    "Alkane",
    "Methyl",
    "Alkene",
    "Alkyne",
    "Alcohols",
    "Amines",
    "Nitriles",
    "Aromatics",
    "Alkyl halides",
    "Esters",
    "Ketones",
    "Aldehydes",
    "Carboxylic acids",
    "Ether",
    "Acyl halides",
    "Amides",
    "Nitro",
)


@dataclass
class FunctionalGroupPredictionTask:
    label_file: str = "functional_groups.npy"
    threshold_search: Optional[dict[str, Any]] = None
    meta_npy_path: Optional[str] = None
    meta_key: str = "functional_group_names_in_order"
    _names: Optional[list[str]] = field(default=None, init=False, repr=False)
    _selected_threshold: Optional[float] = field(default=None, init=False, repr=False)

    @property
    def name(self) -> str:
        return "functional_group_prediction"

    @property
    def label_filename(self) -> str:
        return self.label_file

    def class_names(self) -> Optional[list[str]]:
        if self._names is not None:
            return self._names
        if not self.meta_npy_path:
            return list(FUNCTIONAL_GROUP_NAMES)
        meta = np.load(Path(self.meta_npy_path).expanduser(), allow_pickle=True)
        if isinstance(meta, np.ndarray) and meta.shape == () and meta.dtype == object:
            meta = meta.item()
        if not isinstance(meta, dict) or self.meta_key not in meta:
            raise ValueError(f"Metadata must contain {self.meta_key!r}")
        self._names = [str(value) for value in meta[self.meta_key]]
        return self._names

    @property
    def num_outputs(self) -> int:
        return len(self.class_names() or FUNCTIONAL_GROUP_NAMES)

    def build_criterion(self) -> nn.Module:
        return nn.BCEWithLogitsLoss()

    def forward_model(self, model: nn.Module, signal: torch.Tensor) -> torch.Tensor:
        return model(signal.unsqueeze(1))

    @staticmethod
    def _score(
        probs: np.ndarray,
        targets: np.ndarray,
        threshold: float,
        metric: str = "micro_f1",
    ) -> float:
        prediction = (probs >= threshold).astype(np.int32)
        normalized_metric = metric.lower().replace("-", "_")
        if normalized_metric in {"emr", "exact_match", "exact_match_ratio"}:
            return float(np.mean(np.all(targets == prediction, axis=1)))
        if normalized_metric in {"macro_f1", "macro"}:
            scores = [
                binary_statistics(targets[:, index], prediction[:, index])["f1"]
                for index in range(targets.shape[1])
            ]
            return float(np.mean(scores))
        if normalized_metric not in {"micro_f1", "micro", "f1"}:
            raise ValueError(
                "threshold_search.metric must be micro_f1, macro_f1, or emr; "
                f"got {metric!r}"
            )
        stats = binary_statistics(targets.reshape(-1), prediction.reshape(-1))
        return float(stats["f1"])

    def select_eval_threshold(self, logits: torch.Tensor, targets: torch.Tensor) -> float:
        """Select a threshold on validation outputs for subsequent test scoring."""
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        truth = targets.detach().cpu().numpy().astype(np.int32)
        settings = self.threshold_search or {}
        if settings.get("enabled", True):
            metric = str(settings.get("metric", "micro_f1"))
            candidates = np.linspace(
                float(settings.get("grid_min", 0.05)),
                float(settings.get("grid_max", 0.95)),
                int(settings.get("grid_steps", 19)),
            )
            threshold = float(
                max(candidates, key=lambda value: self._score(probs, truth, value, metric))
            )
        else:
            threshold = float(settings.get("fixed_threshold", 0.5))
        self._selected_threshold = threshold
        return threshold

    @torch.no_grad()
    def eval_from_logits_and_targets(self, logits: torch.Tensor, targets: torch.Tensor):
        probs = torch.sigmoid(logits).cpu().numpy()
        truth = targets.cpu().numpy().astype(np.int32)
        settings = self.threshold_search or {}
        threshold = self._selected_threshold
        if threshold is None:
            threshold = float(settings.get("fixed_threshold", 0.5))
        prediction = (probs >= threshold).astype(np.int32)
        per_class = {
            index: binary_statistics(truth[:, index], prediction[:, index])
            for index in range(truth.shape[1])
        }
        macro = {
            key: float(np.mean([stats[key] for stats in per_class.values()]))
            for key in ("accuracy", "precision", "recall", "f1")
        }
        micro = binary_statistics(truth.reshape(-1), prediction.reshape(-1))
        emr = float(np.mean(np.all(truth == prediction, axis=1)))
        names = self.class_names()
        return {
            "threshold": threshold,
            "overall": {
                # Keep the nested details for compatibility, while exposing the
                # exact paper metric names at the same level as other tasks.
                "micro_f1": float(micro["f1"]),
                "macro_f1": float(macro["f1"]),
                "emr": emr,
                "macro": macro,
                "micro": {key: micro[key] for key in ("accuracy", "precision", "recall", "f1")},
                "subset_accuracy": emr,
                "is_multilabel": True,
            },
            "per_class": per_class,
            "per_class_named": ({names[i]: per_class[i] for i in range(len(names))} if names else {}),
            "class_names": names,
        }
