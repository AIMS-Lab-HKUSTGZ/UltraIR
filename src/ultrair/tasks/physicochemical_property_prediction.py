"""Physicochemical molecular-property prediction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .multioutput_regression import MultioutputRegressionTask


DEFAULT_PROPERTIES = [
    "SAScore", "LogP", "TPSA", "NumHDonors", "NumHAcceptors",
    "NumRotatableBonds", "FractionCSP3", "BertzCT", "QED",
    "NumAromaticRings", "NumAliphaticRings",
]


@dataclass
class PhysicochemicalPropertyPredictionTask(MultioutputRegressionTask):
    task_name: str = "physicochemical_property_prediction"
    label_file: str = "properties.npy"
    property_names: list[str] | None = None

    def __post_init__(self) -> None:
        if self.property_names is None and self.meta_npy_path is None:
            self.property_names = list(DEFAULT_PROPERTIES)
        super().__post_init__()

    @torch.no_grad()
    def eval_from_logits_and_targets(self, predictions: torch.Tensor, targets: torch.Tensor):
        result = super().eval_from_logits_and_targets(predictions, targets)
        if "BertzCT" not in self.property_names:
            return result
        index = self.property_names.index("BertzCT")
        y_pred = self.denormalize(predictions.detach().cpu().numpy())[:, index]
        y_true = self.denormalize(targets.detach().cpu().numpy())[:, index]
        relative_error = np.abs(y_true - y_pred) / (np.abs(y_true) + 1e-10)
        analysis = []
        for threshold in (5, 250, 500, 750, 1000, 1250):
            mask = y_true >= threshold
            if mask.any():
                analysis.append({
                    "threshold": threshold,
                    "count": int(mask.sum()),
                    "mean_rel_err": float(relative_error[mask].mean()),
                })
        result["bertz_analysis"] = analysis
        return result
