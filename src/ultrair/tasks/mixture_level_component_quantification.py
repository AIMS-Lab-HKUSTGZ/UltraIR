"""Mixture-level multi-component quantification with fold-local standardization."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .multioutput_regression import MultioutputRegressionTask


class MixtureSignalPreprocessor:
    def __init__(self, mean: np.ndarray, std: np.ndarray, base=None) -> None:
        self.mean, self.std, self.base = mean.astype(np.float32), std.astype(np.float32), base

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        signal = np.asarray(sample["signal"], dtype=np.float32)
        if signal.ndim == 1:
            signal = signal[None]
        if signal.shape[-1] != self.mean.shape[-1]:
            raise ValueError(
                "Mixture standardization requires the training spectral grid "
                f"with {self.mean.shape[-1]} points; got {signal.shape[-1]}"
            )
        signal = (signal - self.mean[None]) / self.std[None]
        updated = {**sample, "signal": signal.astype(np.float32, copy=False)}
        return self.base(updated) if self.base is not None else updated


@dataclass
class MixtureLevelComponentQuantificationTask(MultioutputRegressionTask):
    task_name: str = "mixture_level_component_quantification"
    label_file: str = "targets.npy"
    component_names_file: str = "component_names.npy"
    property_names: Optional[list[str]] = None
    signal_standardization_eps: float = 1e-6
    disable_default_augmentation: bool = True
    _preprocess: dict[int, tuple[np.ndarray, np.ndarray]] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        # Dataset-local component names and statistics are resolved after the data root is known.
        if self.property_names is not None or self.meta_npy_path or self.labels_npy_path:
            super().__post_init__()

    @property
    def num_outputs(self) -> int:
        if self.property_names is None:
            raise RuntimeError("Component metadata is initialized when building the fold DataLoader")
        return len(self.property_names)

    def configure_data_context(self, cfg, fold: int, train_dir: str, **kwargs) -> None:
        root = Path(cfg["data"]["root"]).expanduser()
        full_data = root / "full_data"
        if not full_data.is_dir():
            full_data = root
        if self.property_names is None:
            names_path = Path(self.meta_npy_path).expanduser() if self.meta_npy_path else full_data / self.component_names_file
            self.property_names = [str(value) for value in np.load(names_path, allow_pickle=True).reshape(-1)]
        if self.stats_mode == "per_fold_train":
            MultioutputRegressionTask.configure_data_context(
                self, cfg=cfg, fold=fold, train_dir=train_dir, **kwargs
            )
        elif not hasattr(self, "center"):
            labels_path = Path(self.labels_npy_path).expanduser() if self.labels_npy_path else full_data / self.label_file
            labels = np.load(labels_path).astype(np.float32)
            if self.target_normalization.lower() in {"standard", "zscore", "z_score"}:
                stats = np.stack([labels.mean(0), labels.std(0)], axis=1)
            else:
                stats = np.stack([labels.min(0), labels.max(0)], axis=1)
            self._set_stats(stats)
        if fold in self._preprocess:
            return
        train_path = Path(
            kwargs.get(
                "train_ir_path",
                Path(train_dir) / cfg["data"].get("ir_name", "ir.npy"),
            )
        )
        try:
            spectra = np.load(train_path, mmap_mode="r")
        except OSError:
            spectra = np.load(train_path)
        if spectra.ndim == 3:
            spectra = spectra[:, 0]
        mean, std = spectra.mean(0), spectra.std(0)
        std = np.maximum(std, self.signal_standardization_eps)
        self._preprocess[fold] = mean, std

    def wrap_transforms(self, cfg, fold: int, train_transform=None, eval_transform=None):
        mean, std = self._preprocess[fold]
        return (
            MixtureSignalPreprocessor(mean, std, train_transform),
            MixtureSignalPreprocessor(mean, std, eval_transform),
        )
