"""Prepare any molecular IR source for the UltraIR molecular tasks.

Source-specific converters converge to ``ir_norm.npy`` + ``smiles.npy``. This
entry point performs the shared steps once: validation, normalization, RDKit
labels, and optional scaffold folds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .aligned_arrays import filter_aligned_arrays
from .molecular_labels import PROPERTY_NAMES, RAW_PATTERNS, generate_labels
from .scaffold_split import scaffold_split
from .spectra import normalize_array, resample_rows


def _string_array(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            value.decode("utf-8", errors="strict")
            if isinstance(value, bytes)
            else str(value)
            for value in values
        ],
        dtype=str,
    )


def prepare(
    ir_path: Path,
    smiles_path: Path,
    output_dir: Path,
    normalize: bool = True,
    make_folds: bool = True,
    k: int = 5,
    seed: int = 42,
    keep_invalid: bool = False,
    valid_fraction: float = 0.1,
    target_points: int = 0,
) -> None:
    ir = np.load(ir_path, allow_pickle=False)
    smiles = np.load(smiles_path, allow_pickle=True)
    if ir.ndim != 2:
        raise ValueError(f"IR must be [N, L], got {ir.shape}")
    if smiles.ndim != 1 or len(smiles) != len(ir):
        raise ValueError(f"IR/SMILES alignment mismatch: {ir.shape[0]} vs {smiles.shape}")
    if target_points < 0:
        raise ValueError("target_points cannot be negative")
    if target_points:
        ir = resample_rows(ir, target_points)
    ir = normalize_array(ir) if normalize else np.asarray(ir, dtype=np.float32)
    if not np.isfinite(ir).all():
        raise ValueError("IR contains non-finite values")
    labels = generate_labels(smiles)
    valid_mask = labels["valid_mask"].astype(bool)
    invalid_count = int(len(valid_mask) - valid_mask.sum())
    arrays: dict[str, np.ndarray] = {
        "ir_norm": ir.astype(np.float32, copy=False),
        "smiles": _string_array(smiles),
        **labels,
    }
    if not keep_invalid:
        if not valid_mask.any():
            raise ValueError("no valid SMILES remain after RDKit validation")
        arrays = filter_aligned_arrays(arrays, valid_mask)
    ir = arrays["ir_norm"]
    folds = (
        scaffold_split(
            arrays["smiles"],
            k=k,
            seed=seed,
            valid_fraction=valid_fraction,
        )
        if make_folds
        else []
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, values in arrays.items():
        np.save(output_dir / f"{name}.npy", values)
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "input_rows": int(len(valid_mask)),
                "output_rows": int(len(ir)),
                "invalid_smiles": invalid_count,
                "invalid_rows_retained": bool(keep_invalid),
                "normalized": bool(normalize),
                "signal_length": int(ir.shape[1]),
                "target_points": int(target_points) if target_points else None,
                "folds": int(k) if make_folds else 0,
                "valid_fraction": float(valid_fraction) if make_folds else None,
                "functional_group_names": [name for name, _ in RAW_PATTERNS],
                "property_names": PROPERTY_NAMES,
                "morgan_radius": 2,
                "morgan_bits": 2048,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if make_folds:
        for fold, (train, valid, test) in enumerate(folds, 1):
            for split, indices in (("train", train), ("valid", valid), ("test", test)):
                split_dir = output_dir / f"fold-{fold}" / split
                split_dir.mkdir(parents=True, exist_ok=True)
                for name, values in arrays.items():
                    np.save(split_dir / f"{name}.npy", values[indices])
            print(f"fold-{fold}: train={len(train)} valid={len(valid)} test={len(test)}")
    print(f"prepared {len(ir)} molecules under {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ir", type=Path, required=True)
    parser.add_argument("--smiles", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--no-folds", action="store_true")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--valid-fraction", type=float, default=0.1)
    parser.add_argument(
        "--target-points",
        type=int,
        default=0,
        help="Resample each spectrum to this common point count before normalization",
    )
    parser.add_argument(
        "--keep-invalid",
        action="store_true",
        help="Retain invalid SMILES rows with empty/NaN labels instead of filtering them",
    )
    args = parser.parse_args()
    prepare(
        args.ir,
        args.smiles,
        args.output_dir,
        not args.no_normalize,
        not args.no_folds,
        args.k,
        args.seed,
        args.keep_invalid,
        args.valid_fraction,
        args.target_points,
    )


if __name__ == "__main__":
    main()
