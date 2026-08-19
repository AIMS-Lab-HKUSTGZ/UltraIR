#!/usr/bin/env python3
"""Convert molecular-dynamics generator CSV output to aligned NumPy arrays."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np


def _header(
    reader: Iterator[list[str]], input_path: Path
) -> tuple[list[str], np.ndarray]:
    try:
        header = next(reader)
    except StopIteration as exc:
        raise ValueError(f"empty CSV: {input_path}") from exc
    if len(header) < 3 or header[0].strip().lower() != "smiles":
        raise ValueError("expected 'smiles' followed by at least two wavenumbers")
    try:
        wavenumbers = np.asarray(
            [float(value) for value in header[1:]], dtype=np.float32
        )
    except ValueError as exc:
        raise ValueError("wavenumber headers must be numeric") from exc
    if not np.isfinite(wavenumbers).all() or np.any(np.diff(wavenumbers) <= 0):
        raise ValueError("wavenumber headers must be finite and strictly ascending")
    return header, wavenumbers


def _row_values(
    row: list[str], header: list[str], input_path: Path, line_number: int
) -> tuple[str, np.ndarray] | None:
    if not row or not any(value.strip() for value in row):
        return None
    if len(row) != len(header):
        raise ValueError(
            f"{input_path}:{line_number}: expected {len(header)} columns, "
            f"got {len(row)}"
        )
    molecule = row[0].strip()
    if not molecule:
        raise ValueError(f"{input_path}:{line_number}: empty SMILES")
    try:
        values = np.asarray([float(value) for value in row[1:]], dtype=np.float32)
    except ValueError as exc:
        raise ValueError(
            f"{input_path}:{line_number}: non-numeric intensity"
        ) from exc
    if not np.isfinite(values).all():
        raise ValueError(f"{input_path}:{line_number}: non-finite intensity")
    return molecule, values


def convert(input_path: Path, output_dir: Path) -> dict[str, object]:
    if not input_path.is_file():
        raise FileNotFoundError(input_path)

    row_count = 0
    max_smiles_length = 1
    with input_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header, wavenumbers = _header(reader, input_path)
        for line_number, row in enumerate(reader, start=2):
            parsed = _row_values(row, header, input_path, line_number)
            if parsed is None:
                continue
            molecule, _ = parsed
            row_count += 1
            max_smiles_length = max(max_smiles_length, len(molecule))

    if row_count == 0:
        raise ValueError(f"CSV contains no spectrum rows: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    ir = np.lib.format.open_memmap(
        output_dir / "ir.npy",
        mode="w+",
        dtype=np.float32,
        shape=(row_count, len(wavenumbers)),
    )
    smiles = np.lib.format.open_memmap(
        output_dir / "smiles.npy",
        mode="w+",
        dtype=f"<U{max_smiles_length}",
        shape=(row_count,),
    )
    with input_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        second_header, _ = _header(reader, input_path)
        output_row = 0
        for line_number, row in enumerate(reader, start=2):
            parsed = _row_values(row, second_header, input_path, line_number)
            if parsed is None:
                continue
            molecule, values = parsed
            smiles[output_row] = molecule
            ir[output_row] = values
            output_row += 1
    ir.flush()
    smiles.flush()
    np.save(output_dir / "wavenumbers.npy", wavenumbers)
    manifest = {
        "source": str(input_path),
        "rows": row_count,
        "signal_length": int(len(wavenumbers)),
        "wavenumber_range_cm-1": [
            float(wavenumbers[0]),
            float(wavenumbers[-1]),
        ],
        "axis_order": "ascending",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"saved {row_count} spectra to {output_dir}")
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    convert(args.input, args.output_dir)


if __name__ == "__main__":
    main()
