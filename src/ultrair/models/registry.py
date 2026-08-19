"""Factory for the UltraIR models used by the paper tasks."""

from __future__ import annotations

from typing import Any

from .pair_classifier import UltraPairClassifier
from .pair_regressor import UltraPairRegressor
from .structural_adapter import UltraIRStructuralAdapter
from .ultrair import UltraIRClassifier


def build_model(name: str, model_cfg: dict[str, Any], dataset_info: Any):
    key = str(name).lower()
    args = dict(model_cfg or {})
    signal_size = int(args.pop("signal_size", dataset_info.signal_size))

    if key in {
        "ultrair",
        "ultrair_classifier",
        "ultrairclassifier",
        "ultrair_pretrained",
        "ultrair_pretrained_classifier",
    }:
        return UltraIRClassifier(
            num_fgroups=int(dataset_info.num_classes),
            signal_size=signal_size,
            **args,
        )
    if key in {"ultrair_structural_adapter", "ultrairstructuraladapter"}:
        return UltraIRStructuralAdapter(signal_size=signal_size, **args)
    if key in {"ultra_pair_classifier", "ultrapairclassifier"}:
        return UltraPairClassifier(signal_size=signal_size, **args)
    if key in {"ultra_pair_regressor", "ultrapairregressor"}:
        return UltraPairRegressor(signal_size=signal_size, **args)
    raise ValueError(f"Unknown UltraIR model: {name}")
