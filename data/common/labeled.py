"""Normalize and split already aligned spectra/labels for non-molecular tasks.

The processor intentionally accepts only NumPy arrays with an established row
alignment. It does not guess spreadsheet columns, identifiers, units, or label
semantics from a provider-specific CSV/Excel file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .spectra import normalize_array, resample_rows
from .splits import train_valid_test_folds


def _load_labels(path: Path) -> np.ndarray:
    labels = np.load(path, allow_pickle=False)
    if labels.ndim not in {1, 2} or labels.shape[0] == 0:
        raise ValueError(f"labels must be non-empty [N] or [N, K], got {labels.shape}")
    if labels.dtype.kind in "fiu" and not np.isfinite(labels).all():
        raise ValueError("labels contain non-finite values")
    if labels.dtype.kind in "SU" and np.any(np.char.strip(labels.astype(str)) == ""):
        raise ValueError("labels contain empty strings")
    return labels


def prepare(
    ir_path: Path,
    labels_path: Path,
    output_dir: Path,
    *,
    ir_name: str = "ir.npy",
    labels_name: str = "labels.npy",
    normalize: bool = True,
    target_points: int = 0,
    k: int = 5,
    seed: int = 42,
    valid_fraction: float = 0.1,
    stratify: bool = False,
) -> dict[str, object]:
    spectra = np.load(ir_path, allow_pickle=False)
    if spectra.ndim != 2 or spectra.shape[0] == 0 or spectra.shape[1] == 0:
        raise ValueError(f"IR must be non-empty [N, L], got {spectra.shape}")
    spectra = np.asarray(spectra, dtype=np.float32)
    if not np.isfinite(spectra).all():
        raise ValueError("IR contains non-finite values")
    labels = _load_labels(labels_path)
    if len(labels) != len(spectra):
        raise ValueError(
            f"IR/label alignment mismatch: {len(spectra)} vs {len(labels)}"
        )
    if target_points < 0:
        raise ValueError("target_points cannot be negative")
    if target_points:
        spectra = resample_rows(spectra, target_points)
    if normalize:
        spectra = normalize_array(spectra)

    stratification = None
    if stratify:
        if labels.ndim != 1:
            raise ValueError("stratified folds require one-dimensional class labels")
        if labels.dtype.kind not in "iu":
            raise ValueError("stratified folds require integer class labels")
        stratification = labels
    folds = train_valid_test_folds(
        len(spectra),
        k=k,
        seed=seed,
        valid_fraction=valid_fraction,
        stratify=stratification,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / ir_name, spectra.astype(np.float32, copy=False))
    np.save(output_dir / labels_name, labels)
    fold_counts: list[dict[str, int]] = []
    for fold, (train, valid, test) in enumerate(folds, start=1):
        counts: dict[str, int] = {}
        for split, indices in (("train", train), ("valid", valid), ("test", test)):
            split_dir = output_dir / f"fold-{fold}" / split
            split_dir.mkdir(parents=True, exist_ok=True)
            np.save(split_dir / ir_name, spectra[indices])
            np.save(split_dir / labels_name, labels[indices])
            counts[split] = int(len(indices))
        fold_counts.append(counts)

    summary: dict[str, object] = {
        "rows": int(len(spectra)),
        "signal_length": int(spectra.shape[1]),
        "label_shape": list(labels.shape[1:]),
        "normalized": bool(normalize),
        "target_points": int(target_points) if target_points else None,
        "folds": int(k),
        "stratified": bool(stratify),
        "seed": int(seed),
        "valid_fraction": float(valid_fraction),
        "fold_counts": fold_counts,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(
        f"prepared spectra={spectra.shape} labels={labels.shape} "
        f"under {output_dir}"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ir", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ir-name", default="ir.npy")
    parser.add_argument("--labels-name", default="labels.npy")
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--target-points", type=int, default=0)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--valid-fraction", type=float, default=0.1)
    parser.add_argument("--stratify", action="store_true")
    args = parser.parse_args()
    prepare(
        args.ir,
        args.labels,
        args.output_dir,
        ir_name=args.ir_name,
        labels_name=args.labels_name,
        normalize=not args.no_normalize,
        target_points=args.target_points,
        k=args.k,
        seed=args.seed,
        valid_fraction=args.valid_fraction,
        stratify=args.stratify,
    )


if __name__ == "__main__":
    main()
