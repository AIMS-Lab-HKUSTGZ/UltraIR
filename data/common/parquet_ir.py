"""Stream local Parquet IR tables into aligned, normalized NumPy arrays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
from contextlib import ExitStack

from .spectra import minmax_normalize
from .array_writer import array_writer


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
        values = []
        for label, name in zip(("id", "smiles", "frequency", "spectrum"), selected):
            column = batch.column(batch.schema.get_field_index(name))
            if label in ("id", "smiles"):
                values.append(column.to_pylist())
            else:
                if column.null_count or column.values.null_count:
                    raise ValueError(f"{label} contains null values")
                if hasattr(column.type, 'list_size'):
                    offsets = (np.arange(len(column) + 1) + column.offset) * column.type.list_size
                else:
                    offsets = column.offsets.to_numpy(zero_copy_only=False)
                flat = column.values.to_numpy(zero_copy_only=False)
                values.append([flat[start:end] for start, end in zip(offsets[:-1], offsets[1:])])
        yield from zip(*values)


def prepare(
    input_dir: Path,
    output_dir: Path,
    pattern: str = "*.parquet",
    target_points: int = 3600,
    low: float = 400.0,
    high: float = 4000.0,
    batch_size: int = 512,
    max_rows: int = 0,
    grid_from_source: bool = False,
    progress: bool = False,
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
    if max_rows < 0:
        raise ValueError("max_rows must be non-negative")

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
    total_rows = min(int(sum(rows_per_file)), max_rows) if max_rows else int(sum(rows_per_file))
    if total_rows == 0:
        raise ValueError("Parquet inputs contain no rows")

    output_dir.mkdir(parents=True, exist_ok=True)
    bounds = (low, high)
    if grid_from_source:
        first = next(_iter_rows(parquet_files[0], schemas[0], 1))
        axis = np.asarray(first[2], dtype=np.float32)
        cropped = np.sort(axis[(axis >= low) & (axis <= high)])
        if len(cropped) < 2:
            raise ValueError("source grid has fewer than two points within crop bounds")
        bounds = (float(cropped[0]), float(cropped[-1]))
    grid = np.linspace(*bounds, target_points, dtype=np.float32)
    ir_path = output_dir / "ir_norm.npy"
    # ExitStack ensures staged output is removed even if conversion fails.
    with ExitStack() as stack:
        spectra = stack.enter_context(array_writer(ir_path, (total_rows, target_points)))
        _convert_rows(parquet_files, schemas, paths, rows_per_file, total_rows, spectra,
                      ids := np.empty(total_rows, dtype=np.int64),
                      smiles := [""] * total_rows, grid, low, high, batch_size, progress)

    np.save(output_dir / "ids.npy", ids)
    np.save(output_dir / "smiles.npy", np.asarray(smiles, dtype=str))
    np.save(output_dir / "wavenumbers.npy", grid)
    manifest = {
        "rows": total_rows,
        "signal_length": target_points,
        "range_cm-1": [low, high],
        "grid_bounds_cm-1": list(bounds),
        "grid_from_source": grid_from_source,
        "axis_order": "ascending",
        "source_files": [path.name for path in paths],
        "rows_per_file": rows_per_file,
        "selected_columns": schemas,
        "max_rows": max_rows or None,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"saved {total_rows} rows to {output_dir}")


def _convert_rows(parquet_files, schemas, paths, rows_per_file, total_rows, spectra,
                  ids, smiles, grid, low, high, batch_size, progress=False):

    from tqdm import tqdm
    bar = tqdm(total=total_rows, desc="[prepare] Spectra", unit="molecule", disable=not progress)
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
            if not np.all(np.diff(axis) > 0):
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
            bar.update(1)
            file_rows += 1
            if cursor == total_rows:
                break
        if cursor == total_rows:
            break
        if file_rows != expected_rows:
            raise RuntimeError(
                f"{path.name}: processed {file_rows} rows, expected {expected_rows}"
            )
    bar.close()
    if cursor != total_rows:
        raise RuntimeError(f"processed {cursor} rows, expected {total_rows}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pattern", default="*.parquet")
    parser.add_argument("--target-points", type=int, default=3600)
    parser.add_argument("--low", type=float, default=400.0)
    parser.add_argument("--high", type=float, default=4000.0)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-rows", type=int, default=0, help="Limit source rows for a smoke run; 0 reads all rows")
    parser.add_argument("--grid-from-source", action="store_true", help="Use cropped source-grid endpoints (legacy USPTO preprocessing)")
    args = parser.parse_args()
    prepare(
        args.input_dir,
        args.output_dir,
        args.pattern,
        args.target_points,
        args.low,
        args.high,
        args.batch_size,
        args.max_rows,
        args.grid_from_source,
    )


if __name__ == "__main__":
    main()
