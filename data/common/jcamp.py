"""Convert local JCAMP-DX files to aligned, traceable NumPy spectra."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from .spectra import normalize_array


NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?")


def _header_value(text: str, name: str) -> str | None:
    match = re.search(rf"^##{re.escape(name)}\s*=\s*(.*?)\s*$", text, re.I | re.M)
    return match.group(1).strip() if match else None


def _header_float(text: str, name: str, default: float | None = None) -> float | None:
    raw = _header_value(text, name)
    if raw is None:
        return default
    match = NUMBER_RE.search(raw)
    if match is None:
        raise ValueError(f"invalid {name} header: {raw!r}")
    return float(match.group(0))


def _xy_step(text: str) -> float:
    delta = _header_float(text, "DELTAX")
    if delta is not None:
        return float(delta)
    first = _header_float(text, "FIRSTX")
    last = _header_float(text, "LASTX")
    points = _header_float(text, "NPOINTS")
    if first is None or last is None or points is None or int(points) < 2:
        raise ValueError("XYDATA requires DELTAX or FIRSTX/LASTX/NPOINTS headers")
    return (float(last) - float(first)) / (int(points) - 1)


def _numeric_line(line: str) -> list[float]:
    # SQZ/DIF/DUP compressed JCAMP uses alphabetic pseudo-digits. Reject those
    # explicitly instead of extracting a plausible but incorrect subset.
    without_exponents = re.sub(r"(?<=\d)[Ee](?=[-+]?\d)", "", line)
    if re.search(r"[A-Za-z@%?]", without_exponents):
        raise ValueError("compressed JCAMP data are not supported")
    return [float(value) for value in NUMBER_RE.findall(line)]


def parse_jcamp(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Parse uncompressed XYDATA or PEAK TABLE data from one JCAMP file."""
    text = path.read_text(encoding="utf-8", errors="replace")
    marker = re.search(r"^##(XYDATA|PEAK TABLE)\s*=.*$", text, re.I | re.M)
    if not marker:
        raise ValueError(f"{path.name}: missing XYDATA/PEAK TABLE")
    factor_x = float(_header_float(text, "XFACTOR", 1.0))
    factor_y = float(_header_float(text, "YFACTOR", 1.0))
    is_peak_table = marker.group(1).upper() == "PEAK TABLE"
    delta_x = None if is_peak_table else _xy_step(text)
    points: list[tuple[float, float]] = []
    for line in text[marker.end():].splitlines():
        if line.lstrip().startswith("##"):
            break
        if not line.strip():
            continue
        values = _numeric_line(line)
        if len(values) < 2:
            continue
        if is_peak_table:
            if len(values) % 2:
                raise ValueError(f"{path.name}: odd number of PEAK TABLE values")
            points.extend(
                (values[index] * factor_x, values[index + 1] * factor_y)
                for index in range(0, len(values), 2)
            )
        else:
            start_x = values[0] * factor_x
            points.extend(
                (start_x + index * float(delta_x), value * factor_y)
                for index, value in enumerate(values[1:])
            )
    if len(points) < 2:
        raise ValueError(f"{path.name}: no spectrum points")
    values = np.asarray(points, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"{path.name}: spectrum contains non-finite values")
    order = np.argsort(values[:, 0], kind="stable")
    sorted_axis = values[order, 0]
    sorted_signal = values[order, 1]
    axis, unique_indices = np.unique(sorted_axis, return_index=True)
    signal = sorted_signal[unique_indices]
    if len(axis) < 2:
        raise ValueError(f"{path.name}: fewer than two unique x values")
    return axis.astype(np.float32), signal.astype(np.float32)


def _source_identifier(path: Path, text: str) -> str:
    for name in ("CAS REGISTRY NO", "CAS", "NIST CHEMISTRY WEBBOOK ID"):
        value = _header_value(text, name)
        if value:
            return value
    return path.stem


def _load_smiles(smiles_dir: Path, stem: str) -> str:
    path = smiles_dir / f"{stem}.txt"
    if not path.is_file():
        raise FileNotFoundError(path)
    values = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(values) != 1:
        raise ValueError(f"{path} must contain exactly one non-empty SMILES line")
    return values[0]


def prepare(
    input_dir: Path,
    output_dir: Path,
    target_points: int = 3600,
    low: float = 400.0,
    high: float = 4000.0,
    smiles_dir: Path | None = None,
    skip_invalid: bool = False,
) -> None:
    if target_points < 2:
        raise ValueError("target_points must be at least 2")
    if not np.isfinite([low, high]).all() or low >= high:
        raise ValueError("expected finite wavenumber bounds with low < high")
    paths = sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jdx", ".dx"}
    )
    if not paths:
        raise FileNotFoundError(f"no .jdx or .dx files under {input_dir}")

    rows: list[np.ndarray] = []
    source_ids: list[str] = []
    file_names: list[str] = []
    smiles: list[str] = []
    rejected: list[dict[str, str]] = []
    grid = np.linspace(low, high, target_points, dtype=np.float32)
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
            axis, signal = parse_jcamp(path)
            mask = (axis >= low) & (axis <= high)
            if np.count_nonzero(mask) < 2:
                raise ValueError(
                    f"{path.name}: fewer than two points in [{low}, {high}] cm^-1"
                )
            row = np.interp(grid, axis[mask], signal[mask]).astype(np.float32)
            row_smiles = _load_smiles(smiles_dir, path.stem) if smiles_dir else None
        except (OSError, UnicodeError, ValueError) as exc:
            if not skip_invalid:
                raise
            rejected.append({"file": path.name, "error": str(exc)})
            continue
        rows.append(row)
        source_ids.append(_source_identifier(path, text))
        file_names.append(path.name)
        if row_smiles is not None:
            smiles.append(row_smiles)
    if not rows:
        raise RuntimeError(f"no valid JCAMP files under {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "ir_norm.npy", normalize_array(np.stack(rows)))
    np.save(output_dir / "wavenumbers.npy", grid)
    np.save(output_dir / "source_ids.npy", np.asarray(source_ids, dtype=str))
    np.save(output_dir / "source_files.npy", np.asarray(file_names, dtype=str))
    if smiles_dir is not None:
        np.save(output_dir / "smiles.npy", np.asarray(smiles, dtype=str))
    manifest = {
        "rows": len(rows),
        "rejected_rows": len(rejected),
        "signal_length": target_points,
        "range_cm-1": [low, high],
        "axis_order": "ascending",
        "smiles_aligned": smiles_dir is not None,
        "rejected": rejected,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"saved {len(rows)} spectra to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-points", type=int, default=3600)
    parser.add_argument("--low", type=float, default=400.0)
    parser.add_argument("--high", type=float, default=4000.0)
    parser.add_argument("--smiles-dir", type=Path)
    parser.add_argument("--skip-invalid", action="store_true")
    args = parser.parse_args()
    prepare(
        args.input_dir,
        args.output_dir,
        args.target_points,
        args.low,
        args.high,
        args.smiles_dir,
        args.skip_invalid,
    )


if __name__ == "__main__":
    main()
