#!/usr/bin/env python3
"""Average two official Chemprop-IR prediction CSV files.

The external repository produces ordinary CSV files; this script validates
their row and target-column alignment, averages the two spectra, and writes
``ir.npy`` plus ``smiles.npy`` for ``data.common.molecular``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


IGNORED_COLUMNS = {"compound_names", "epi_unc"}


def _target_columns(header: list[str], smiles_index: int) -> list[int]:
    columns: list[int] = []
    for index, name in enumerate(header):
        normalized = name.strip().lower()
        if index == smiles_index or normalized in IGNORED_COLUMNS or normalized.endswith("_epi_unc"):
            continue
        columns.append(index)
    if not columns:
        raise ValueError("prediction CSV has no intensity columns")
    names = [header[index].strip() for index in columns]
    if any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError("prediction CSV contains empty or duplicate intensity headers")
    return columns


def read_predictions(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Read one Chemprop-IR prediction CSV as ``(smiles, spectra, headers)``."""
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"prediction CSV is empty: {path}") from exc
        normalized = [name.strip().lower() for name in header]
        if normalized.count("smiles") != 1:
            raise ValueError(f"{path}: expected exactly one 'smiles' column")
        smiles_index = normalized.index("smiles")
        target_indices = _target_columns(header, smiles_index)
        smiles: list[str] = []
        spectra: list[list[float]] = []
        expected_width = len(header)
        for line_number, row in enumerate(reader, start=2):
            if not row or not any(cell.strip() for cell in row):
                continue
            if len(row) != expected_width:
                raise ValueError(
                    f"{path}:{line_number}: expected {expected_width} columns, got {len(row)}"
                )
            text = row[smiles_index].strip()
            if not text:
                raise ValueError(f"{path}:{line_number}: empty SMILES")
            values: list[float] = []
            for index in target_indices:
                raw = row[index].strip()
                if not raw:
                    raise ValueError(f"{path}:{line_number}: empty intensity value")
                try:
                    value = float(raw)
                except ValueError as exc:
                    raise ValueError(
                        f"{path}:{line_number}: non-numeric intensity {raw!r}"
                    ) from exc
                if not np.isfinite(value):
                    raise ValueError(f"{path}:{line_number}: non-finite intensity")
                values.append(value)
            smiles.append(text)
            spectra.append(values)
    if not spectra:
        raise ValueError(f"prediction CSV contains no data rows: {path}")
    return (
        np.asarray(smiles, dtype=str),
        np.asarray(spectra, dtype=np.float32),
        [header[index].strip() for index in target_indices],
    )


def combine(
    prediction_paths: tuple[Path, Path],
    output_dir: Path,
) -> dict[str, object]:
    """Validate and equally average two prediction files."""
    first_smiles, first_spectra, headers = read_predictions(prediction_paths[0])
    second_smiles, second_spectra, second_headers = read_predictions(prediction_paths[1])
    if headers != second_headers:
        raise ValueError("prediction files do not have identical intensity headers")
    if first_smiles.shape != second_smiles.shape or not np.array_equal(first_smiles, second_smiles):
        raise ValueError("prediction files do not have identical SMILES rows in identical order")
    if first_spectra.shape != second_spectra.shape:
        raise ValueError("prediction files do not have identical spectrum shapes")
    averaged = ((first_spectra.astype(np.float64) + second_spectra.astype(np.float64)) / 2.0).astype(np.float32)
    if not np.isfinite(averaged).all():
        raise ValueError("averaged predictions contain non-finite values")
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "ir.npy", averaged)
    np.save(output_dir / "smiles.npy", first_smiles)
    manifest = {
        "rows": int(len(first_smiles)),
        "points": int(averaged.shape[1]),
        "intensity_headers": headers,
        "combination": "equal arithmetic mean",
        "prediction_files": [str(path) for path in prediction_paths],
        "normalized": False,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"saved {len(first_smiles)} spectra with {averaged.shape[1]} points to {output_dir}")
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        type=Path,
        nargs=2,
        metavar=("MODEL_A_CSV", "MODEL_B_CSV"),
        required=True,
        help="Two CSVs produced by the official Chemprop-IR predict.py",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    combine((args.predictions[0], args.predictions[1]), args.output_dir)


if __name__ == "__main__":
    main()
