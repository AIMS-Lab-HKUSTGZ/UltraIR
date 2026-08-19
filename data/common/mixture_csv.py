"""Prepare a local spectra CSV plus target CSV for mixture quantification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .spectra import minmax_normalize, resample_rows


def _read_numeric(path: Path) -> np.ndarray:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("CSV mixture processing requires pandas") from exc
    frame = pd.read_csv(path, header=None).apply(pd.to_numeric, errors="coerce")
    frame = frame.dropna(axis=0, how="all").dropna(axis=1, how="all")
    if frame.empty:
        raise ValueError(f"no numeric values in {path}")
    values = frame.to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        row, column = np.argwhere(~np.isfinite(values))[0]
        raise ValueError(
            f"non-numeric or missing value in {path} at numeric row {row}, column {column}"
        )
    return values


def prepare(spectra_csv: Path, targets_csv: Path, output_dir: Path,
            target_points: int = 0, normalize: bool = True) -> None:
    if target_points < 0:
        raise ValueError("target_points cannot be negative")
    spectra = _read_numeric(spectra_csv)
    targets = _read_numeric(targets_csv)
    wavenumbers = None
    # Some public tables include one wavenumber row before the samples.
    if spectra.shape[0] == targets.shape[0] + 1:
        wavenumbers = spectra[0].copy()
        spectra = spectra[1:]
    if spectra.ndim != 2 or targets.ndim != 2 or spectra.shape[0] != targets.shape[0]:
        raise ValueError(f"unaligned CSVs: spectra={spectra.shape}, targets={targets.shape}")
    if target_points:
        if wavenumbers is None:
            spectra = resample_rows(spectra, target_points)
        else:
            order = np.argsort(wavenumbers)
            axis = wavenumbers[order]
            axis, unique = np.unique(axis, return_index=True)
            if len(axis) < 2:
                raise ValueError("wavenumber row must contain at least two distinct values")
            grid = np.linspace(float(axis[0]), float(axis[-1]), target_points, dtype=np.float32)
            spectra = np.stack(
                [np.interp(grid, axis, row[order][unique]) for row in spectra],
                axis=0,
            ).astype(np.float32)
            wavenumbers = grid
    if normalize:
        spectra = np.stack([minmax_normalize(row) for row in spectra], axis=0)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "spectra.npy", spectra.astype(np.float32))
    np.save(output_dir / "targets.npy", targets.astype(np.float32))
    if wavenumbers is not None:
        np.save(output_dir / "wavenumbers.npy", wavenumbers.astype(np.float32))
    (output_dir / "manifest.json").write_text(json.dumps({
        "samples": len(spectra),
        "signal_length": spectra.shape[1],
        "targets": targets.shape[1],
        "normalized": bool(normalize),
        "has_wavenumbers": wavenumbers is not None,
    }, indent=2), encoding="utf-8")
    print(f"saved spectra={spectra.shape} targets={targets.shape} to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spectra-csv", type=Path, required=True)
    parser.add_argument("--targets-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-points", type=int, default=0)
    parser.add_argument("--no-normalize", action="store_true")
    args = parser.parse_args()
    prepare(
        args.spectra_csv,
        args.targets_csv,
        args.output_dir,
        args.target_points,
        not args.no_normalize,
    )


if __name__ == "__main__":
    main()
