"""Small, dependency-light helpers for IR array preparation.

The old dataset scripts each implemented a slightly different version of this
logic.  These functions deliberately operate on arrays and leave file layout
decisions to the dataset/task entry points.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _finite_float_array(values: np.ndarray, *, ndim: int, label: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32)
    if result.ndim != ndim:
        raise ValueError(f"expected {label} with {ndim} dimensions, got {result.shape}")
    if result.size == 0:
        raise ValueError(f"{label} is empty")
    if not np.isfinite(result).all():
        bad = int(result.size - np.count_nonzero(np.isfinite(result)))
        raise ValueError(f"{label} contains {bad} non-finite values")
    return result


def minmax_normalize(row: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Normalize one finite spectrum to [0, 1]."""
    values = _finite_float_array(row, ndim=1, label="spectrum")
    lo = float(values.min())
    hi = float(values.max())
    if hi - lo <= eps:
        return np.zeros_like(values)
    return ((values - lo) / (hi - lo)).astype(np.float32, copy=False)


def normalize_array(spectra: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Row-wise min-max normalization for a [N, L] array."""
    values = _finite_float_array(spectra, ndim=2, label="spectra [N, L]")
    return np.stack([minmax_normalize(row, eps=eps) for row in values], axis=0)


def resample_rows(spectra: np.ndarray, target_points: int) -> np.ndarray:
    """Linearly resample [N, L] spectra to a common number of points."""
    values = _finite_float_array(spectra, ndim=2, label="spectra [N, L]")
    if target_points < 1:
        raise ValueError(f"target_points must be positive, got {target_points}")
    if values.shape[1] == target_points:
        return values.copy()
    old_x = np.linspace(0.0, 1.0, values.shape[1], dtype=np.float64)
    new_x = np.linspace(0.0, 1.0, target_points, dtype=np.float64)
    return np.stack([np.interp(new_x, old_x, row).astype(np.float32) for row in values], axis=0)


def _main() -> None:
    parser = argparse.ArgumentParser(description="Normalize an IR .npy array row by row.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-points", type=int, default=0)
    args = parser.parse_args()
    values = np.load(args.input, allow_pickle=False)
    values = resample_rows(values, args.target_points) if args.target_points else np.asarray(values)
    result = normalize_array(values)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, result)
    print(f"saved {args.output}: shape={result.shape} dtype={result.dtype}")


if __name__ == "__main__":
    _main()
