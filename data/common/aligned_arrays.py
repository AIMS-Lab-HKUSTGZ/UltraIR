"""Validate, filter, and reproducibly subset aligned NumPy arrays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

import numpy as np


DEFAULT_STATIC_ARRAYS = {
    "wavenumbers.npy",
    "component_names.npy",
    "pure_component_spectra.npy",
    "pure_component_names.npy",
}


def validate_mask(mask: np.ndarray, expected_rows: int | None = None) -> np.ndarray:
    values = np.asarray(mask)
    if values.ndim != 1:
        raise ValueError(f"mask must be 1D, got {values.shape}")
    if expected_rows is not None and len(values) != expected_rows:
        raise ValueError(f"mask has {len(values)} rows, expected {expected_rows}")
    if values.dtype == np.bool_:
        return values
    if not np.issubdtype(values.dtype, np.integer) or not np.isin(values, [0, 1]).all():
        raise ValueError("mask must be boolean or contain only integer 0/1 values")
    return values.astype(bool)


def filter_aligned_arrays(
    arrays: Mapping[str, np.ndarray], mask: np.ndarray
) -> dict[str, np.ndarray]:
    selected = validate_mask(mask)
    output: dict[str, np.ndarray] = {}
    for name, array in arrays.items():
        values = np.asarray(array)
        if values.ndim == 0 or values.shape[0] != len(selected):
            raise ValueError(
                f"{name}: expected first dimension {len(selected)}, got {values.shape}"
            )
        output[name] = values[selected]
    return output


def _load_directory(
    input_dir: Path,
    reference_name: str,
    static_names: set[str],
) -> tuple[int, dict[str, np.ndarray], dict[str, np.ndarray]]:
    reference_path = input_dir / reference_name
    if not reference_path.is_file():
        raise FileNotFoundError(reference_path)
    reference = np.load(reference_path, allow_pickle=True)
    if reference.ndim == 0:
        raise ValueError(f"{reference_path} is scalar")
    n_rows = int(reference.shape[0])
    aligned: dict[str, np.ndarray] = {}
    static: dict[str, np.ndarray] = {}
    for path in sorted(input_dir.glob("*.npy")):
        values = np.load(path, allow_pickle=True)
        if values.ndim > 0 and values.shape[0] == n_rows:
            aligned[path.name] = values
        elif path.name in static_names:
            static[path.name] = values
        else:
            raise ValueError(
                f"{path.name}: shape {values.shape} is neither aligned to {n_rows} rows "
                "nor declared static"
            )
    return n_rows, aligned, static


def _write_directory(
    output_dir: Path,
    aligned: Mapping[str, np.ndarray],
    static: Mapping[str, np.ndarray],
    indices: np.ndarray,
    operation: dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, values in aligned.items():
        np.save(output_dir / name, values[indices])
    for name, values in static.items():
        np.save(output_dir / name, values)
    source_indices = (
        np.asarray(aligned["source_indices.npy"])[indices]
        if "source_indices.npy" in aligned
        else indices
    )
    np.save(
        output_dir / "source_indices.npy",
        source_indices.astype(np.int64, copy=False),
    )
    (output_dir / "alignment_manifest.json").write_text(
        json.dumps(operation, indent=2), encoding="utf-8"
    )


def filter_directory(
    input_dir: Path,
    output_dir: Path,
    mask_name: str = "valid_mask.npy",
    reference_name: str = "smiles.npy",
    static_names: set[str] | None = None,
) -> np.ndarray:
    if input_dir.resolve() == output_dir.resolve():
        raise ValueError("input_dir and output_dir must be different")
    n_rows, aligned, static = _load_directory(
        input_dir, reference_name, static_names or DEFAULT_STATIC_ARRAYS
    )
    if mask_name not in aligned:
        raise FileNotFoundError(input_dir / mask_name)
    mask = validate_mask(aligned[mask_name], n_rows)
    indices = np.flatnonzero(mask).astype(np.int64)
    if not len(indices):
        raise ValueError("mask selects no rows")
    _write_directory(
        output_dir,
        aligned,
        static,
        indices,
        {
            "operation": "filter",
            "input_rows": n_rows,
            "output_rows": int(len(indices)),
            "mask": mask_name,
            "reference": reference_name,
        },
    )
    return indices


def subset_directory(
    input_dir: Path,
    output_dir: Path,
    ratio: float,
    seed: int = 42,
    reference_name: str = "smiles.npy",
    static_names: set[str] | None = None,
) -> np.ndarray:
    if input_dir.resolve() == output_dir.resolve():
        raise ValueError("input_dir and output_dir must be different")
    if not 0.0 < ratio <= 1.0:
        raise ValueError("ratio must be in (0, 1]")
    n_rows, aligned, static = _load_directory(
        input_dir, reference_name, static_names or DEFAULT_STATIC_ARRAYS
    )
    count = max(1, int(round(n_rows * ratio)))
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(n_rows, size=count, replace=False)).astype(np.int64)
    _write_directory(
        output_dir,
        aligned,
        static,
        indices,
        {
            "operation": "subset",
            "input_rows": n_rows,
            "output_rows": int(len(indices)),
            "ratio": float(ratio),
            "seed": int(seed),
            "reference": reference_name,
        },
    )
    return indices


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="operation", required=True)

    filter_parser = subparsers.add_parser("filter")
    filter_parser.add_argument("--input-dir", type=Path, required=True)
    filter_parser.add_argument("--output-dir", type=Path, required=True)
    filter_parser.add_argument("--mask", default="valid_mask.npy")
    filter_parser.add_argument("--reference", default="smiles.npy")

    subset_parser = subparsers.add_parser("subset")
    subset_parser.add_argument("--input-dir", type=Path, required=True)
    subset_parser.add_argument("--output-dir", type=Path, required=True)
    subset_parser.add_argument("--ratio", type=float, required=True)
    subset_parser.add_argument("--seed", type=int, default=42)
    subset_parser.add_argument("--reference", default="smiles.npy")

    args = parser.parse_args()
    if args.operation == "filter":
        indices = filter_directory(
            args.input_dir, args.output_dir, args.mask, args.reference
        )
    else:
        indices = subset_directory(
            args.input_dir, args.output_dir, args.ratio, args.seed, args.reference
        )
    print(f"saved {len(indices)} aligned rows to {args.output_dir}")


if __name__ == "__main__":
    main()
