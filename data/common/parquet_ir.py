"""Stream local Parquet IR tables into aligned, normalized NumPy arrays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
from numpy.lib.format import open_memmap

from .spectra import minmax_normalize


COLUMN_CANDIDATES = {
    "id": ("id", "ID", "cid", "CID"),
    "smiles": ("SMILES", "smiles", "canonical_smiles"),
    "frequency": ("Frequency(cm^-1)", "Frequency", "frequency", "wavenumbers"),
    "spectrum": ("ir_spectra", "IR_spectra", "spectra", "spectrum"),
}


def _column(names: Sequence[str], candidates: Sequence[str], label: str) -> str:
    for name in candidates:
        if name in names:
            return name
    raise KeyError(f"missing {label}; available columns={list(names)}")


def _schema(names: Sequence[str]) -> dict[str, str]:
    return {
        label: _column(names, candidates, label)
        for label, candidates in COLUMN_CANDIDATES.items()
    }


def _finite_vector(values: object, *, label: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or result.size < 2:
        raise ValueError(f"{label} must be a 1D array with at least two values")
    if not np.isfinite(result).all():
        raise ValueError(f"{label} contains non-finite values")
    return result


def _iter_rows(
    parquet_file: object,
    columns: dict[str, str],
    batch_size: int,
) -> Iterator[tuple[object, object, object, object]]:
    selected = [columns[name] for name in ("id", "smiles", "frequency", "spectrum")]
    for batch in parquet_file.iter_batches(columns=selected, batch_size=batch_size):
        values = [
            batch.column(batch.schema.get_field_index(name)).to_pylist()
            for name in selected
        ]
        yield from zip(*values)


def prepare(
    input_dir: Path,
    output_dir: Path,
    pattern: str = "*.parquet",
    target_points: int = 3600,
    low: float = 400.0,
    high: float = 4000.0,
    batch_size: int = 512,
) -> None:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("Parquet processing requires pyarrow") from exc
    if target_points < 2:
        raise ValueError("target_points must be at least 2")
    if not np.isfinite([low, high]).all() or low >= high:
        raise ValueError("expected finite wavenumber bounds with low < high")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    paths = sorted(input_dir.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"no files matched {pattern!r} under {input_dir}")

    parquet_files = []
    schemas = []
    rows_per_file = []
    for path in paths:
        parquet_file = pq.ParquetFile(path)
        parquet_files.append(parquet_file)
        schemas.append(_schema(parquet_file.schema_arrow.names))
        rows_per_file.append(int(parquet_file.metadata.num_rows))
    total_rows = int(sum(rows_per_file))
    if total_rows == 0:
        raise ValueError("Parquet inputs contain no rows")

    output_dir.mkdir(parents=True, exist_ok=True)
    grid = np.linspace(low, high, target_points, dtype=np.float32)
    ir_path = output_dir / "ir_norm.npy"
    spectra = open_memmap(
        ir_path,
        mode="w+",
        dtype=np.float32,
        shape=(total_rows, target_points),
    )
    ids = np.empty(total_rows, dtype=np.int64)
    smiles: list[str] = [""] * total_rows

    cursor = 0
    for path, parquet_file, columns, expected_rows in zip(
        paths, parquet_files, schemas, rows_per_file
    ):
        file_rows = 0
        for raw_id, raw_smiles, raw_axis, raw_signal in _iter_rows(
            parquet_file, columns, batch_size
        ):
            row_label = f"{path.name}:row-{file_rows}"
            if raw_id is None:
                raise ValueError(f"{row_label} has a null id")
            try:
                molecule_id = int(raw_id)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{row_label} has non-integer id {raw_id!r}") from exc
            text = "" if raw_smiles is None else str(raw_smiles).strip()
            if not text:
                raise ValueError(f"{row_label} has an empty SMILES")
            axis = _finite_vector(raw_axis, label=f"{row_label} frequency")
            signal = _finite_vector(raw_signal, label=f"{row_label} spectrum")
            if axis.shape != signal.shape:
                raise ValueError(
                    f"{row_label} frequency/spectrum mismatch: {axis.shape} vs {signal.shape}"
                )
            order = np.argsort(axis, kind="stable")
            axis = axis[order]
            signal = signal[order]
            axis, unique_indices = np.unique(axis, return_index=True)
            signal = signal[unique_indices]
            mask = (axis >= low) & (axis <= high)
            if np.count_nonzero(mask) < 2:
                raise ValueError(
                    f"{row_label} has fewer than two points in [{low}, {high}] cm^-1"
                )
            interpolated = np.interp(grid, axis[mask], signal[mask]).astype(np.float32)
            spectra[cursor] = minmax_normalize(interpolated)
            ids[cursor] = molecule_id
            smiles[cursor] = text
            cursor += 1
            file_rows += 1
        if file_rows != expected_rows:
            raise RuntimeError(
                f"{path.name}: processed {file_rows} rows, expected {expected_rows}"
            )
    if cursor != total_rows:
        raise RuntimeError(f"processed {cursor} rows, expected {total_rows}")
    spectra.flush()

    np.save(output_dir / "ids.npy", ids)
    np.save(output_dir / "smiles.npy", np.asarray(smiles, dtype=str))
    np.save(output_dir / "wavenumbers.npy", grid)
    manifest = {
        "rows": total_rows,
        "signal_length": target_points,
        "range_cm-1": [low, high],
        "axis_order": "ascending",
        "source_files": [path.name for path in paths],
        "rows_per_file": rows_per_file,
        "selected_columns": schemas,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"saved {total_rows} rows to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pattern", default="*.parquet")
    parser.add_argument("--target-points", type=int, default=3600)
    parser.add_argument("--low", type=float, default=400.0)
    parser.add_argument("--high", type=float, default=4000.0)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()
    prepare(
        args.input_dir,
        args.output_dir,
        args.pattern,
        args.target_points,
        args.low,
        args.high,
        args.batch_size,
    )


if __name__ == "__main__":
    main()
