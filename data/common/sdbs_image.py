"""Digitize SDBS PNG spectrum plots into traceable NumPy arrays.

SDBS uses a two-scale wavenumber axis: 4000--2000 cm^-1 occupies the
first 10/26 of the plot and 2000--400 cm^-1 occupies the remaining 16/26.
This converter handles that source format only; it never fetches molecular
metadata from external services.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np


PHASE_ALIASES = {
    "ccl4": "ccl4",
    "carbon tetrachloride": "ccl4",
    "kbr": "kbr",
    "potassium bromide": "kbr",
    "liquid": "liquid",
    "nujol": "nujolmull",
    "nujol mull": "nujolmull",
    "nujolmull": "nujolmull",
}


def _load_grayscale(path: Path) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("SDBS PNG conversion requires Pillow") from exc

    try:
        with Image.open(path) as image:
            values = np.asarray(image.convert("L"), dtype=np.uint8)
    except OSError as exc:
        raise ValueError(f"cannot read image {path}: {exc}") from exc
    if values.ndim != 2 or min(values.shape) < 10:
        raise ValueError(f"invalid grayscale image shape for {path}: {values.shape}")
    return values


def _longest_run(row: np.ndarray) -> tuple[int, int, int] | None:
    padded = np.concatenate(([False], np.asarray(row, dtype=bool), [False]))
    transitions = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(transitions == 1)
    ends = np.flatnonzero(transitions == -1) - 1
    if not len(starts):
        return None
    lengths = ends - starts + 1
    best = int(np.argmax(lengths))
    return int(starts[best]), int(ends[best]), int(lengths[best])


def find_plot_bounds(
    grayscale: np.ndarray,
    *,
    threshold: int = 160,
) -> tuple[int, int, int, int]:
    """Return inclusive ``(left, top, right, bottom)`` SDBS plot borders."""
    image = np.asarray(grayscale)
    if image.ndim != 2:
        raise ValueError(f"expected a 2D grayscale image, got {image.shape}")
    if not 0 <= threshold <= 255:
        raise ValueError(f"threshold must be in [0, 255], got {threshold}")

    height, width = image.shape
    dark = image <= threshold
    rows: list[tuple[int, int, int]] = []
    minimum_run = max(int(np.ceil(width * 0.70)), 10)
    for y, row in enumerate(dark):
        run = _longest_run(row)
        if run is not None and run[2] >= minimum_run:
            rows.append((y, run[0], run[1]))

    candidates: list[tuple[float, int, int, int, int]] = []
    minimum_height = max(int(np.ceil(height * 0.35)), 10)
    maximum_height = int(np.floor(height * 0.72))
    edge_window = max(int(np.ceil(width * 0.04)), 8)
    for top_index, (top, top_left, top_right) in enumerate(rows):
        for bottom, bottom_left, bottom_right in rows[top_index + 1 :]:
            plot_height = bottom - top
            if plot_height < minimum_height or plot_height > maximum_height:
                continue

            left_start = max(min(top_left, bottom_left) - 2, 0)
            left_stop = min(max(top_left, bottom_left) + edge_window + 1, width)
            right_start = max(min(top_right, bottom_right) - edge_window, 0)
            right_stop = min(max(top_right, bottom_right) + 3, width)
            vertical = dark[top : bottom + 1]
            left_density = vertical[:, left_start:left_stop].mean(axis=0)
            right_density = vertical[:, right_start:right_stop].mean(axis=0)
            left_offset = int(np.argmax(left_density))
            right_offset = int(np.argmax(right_density))
            left = left_start + left_offset
            right = right_start + right_offset
            edge_score = min(
                float(left_density[left_offset]), float(right_density[right_offset])
            )
            if edge_score < 0.90 or right - left < minimum_run:
                continue
            candidates.append((edge_score, top, bottom, left, right))

    if not candidates:
        raise ValueError("could not locate the bordered SDBS spectrum plot")
    _, top, bottom, left, right = max(
        candidates,
        key=lambda item: (item[0], item[4] - item[3], item[2] - item[1]),
    )
    return left, top, right, bottom


def _trace_curve(plot: np.ndarray, *, threshold: int) -> np.ndarray:
    dark = np.asarray(plot) <= threshold
    curve = np.full(dark.shape[1], np.nan, dtype=np.float64)
    for column in range(dark.shape[1]):
        y_values = np.flatnonzero(dark[:, column])
        if len(y_values):
            curve[column] = float(np.median(y_values))
    valid = np.isfinite(curve)
    if np.count_nonzero(valid) < max(2, int(np.ceil(len(curve) * 0.50))):
        raise ValueError("too few curve pixels inside the SDBS plot")
    pixels = np.arange(len(curve), dtype=np.float64)
    return np.interp(pixels, pixels[valid], curve[valid])


def extract_spectrum(
    image_path: Path,
    *,
    target_points: int = 3600,
    low: float = 400.0,
    high: float = 4000.0,
    threshold: int = 160,
) -> np.ndarray:
    """Extract one SDBS transmittance curve on an ascending wavenumber grid."""
    if target_points < 2:
        raise ValueError("target_points must be at least 2")
    if not np.isfinite([low, high]).all() or low >= high:
        raise ValueError("expected finite wavenumber bounds with low < high")
    if low < 400.0 or high > 4000.0:
        raise ValueError("SDBS image calibration only covers 400--4000 cm^-1")

    image = _load_grayscale(image_path)
    left, top, right, bottom = find_plot_bounds(image, threshold=threshold)
    plot = image[top + 1 : bottom, left + 1 : right]
    if min(plot.shape) < 2:
        raise ValueError(f"empty plot interior in {image_path}")
    y_pixels = _trace_curve(plot, threshold=threshold)
    transmittance = 1.0 - y_pixels / max(plot.shape[0] - 1, 1)

    x_pixels = np.arange(plot.shape[1], dtype=np.float64)
    full_span = max(plot.shape[1] - 1, 1)
    break_pixel = full_span * (10.0 / 26.0)
    wavenumbers_desc = np.where(
        x_pixels <= break_pixel,
        4000.0 - (2000.0 / break_pixel) * x_pixels,
        2000.0 - (1600.0 / (full_span - break_pixel)) * (x_pixels - break_pixel),
    )
    grid = np.linspace(low, high, target_points, dtype=np.float64)
    result = np.interp(grid, wavenumbers_desc[::-1], transmittance[::-1])
    result = np.clip(result, 0.0, 1.0).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError(f"non-finite spectrum extracted from {image_path}")
    return result


def normalize_phase(value: str) -> str | None:
    token = re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()
    if token in PHASE_ALIASES:
        return PHASE_ALIASES[token]
    compact = token.replace(" ", "")
    for alias, phase in PHASE_ALIASES.items():
        if alias.replace(" ", "") in compact:
            return phase
    return None


def detect_phase(image_path: Path) -> str | None:
    """Read the phase label in the SDBS image header with optional OCR."""
    try:
        import pytesseract
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("phase OCR requires Pillow and pytesseract") from exc

    with Image.open(image_path) as image:
        grayscale = image.convert("L")
        width, height = grayscale.size
        header = grayscale.crop((int(width * 0.52), 0, width, max(int(height * 0.25), 1)))
        header = header.resize((header.width * 2, header.height * 2))
        text = pytesseract.image_to_string(header, config="--psm 6")
    return normalize_phase(text)


def _clean_required(row: dict[str, str | None], field: str, row_number: int) -> str:
    value = (row.get(field) or "").strip()
    if not value:
        raise ValueError(f"CSV row {row_number}: missing {field!r}")
    return value


def prepare(
    metadata_csv: Path,
    output_dir: Path,
    *,
    image_root: Path | None = None,
    target_points: int = 3600,
    low: float = 400.0,
    high: float = 4000.0,
    threshold: int = 160,
    image_field: str = "Image File",
    id_field: str = "SDBS No",
    smiles_field: str | None = None,
    phase_field: str | None = None,
    detect_phase_with_ocr: bool = False,
    skip_invalid: bool = False,
) -> None:
    """Convert images referenced by one SDBS metadata CSV."""
    if not metadata_csv.is_file():
        raise FileNotFoundError(metadata_csv)
    root = image_root if image_root is not None else metadata_csv.parent

    with metadata_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        required = [image_field, id_field]
        if smiles_field:
            required.append(smiles_field)
        if phase_field:
            required.append(phase_field)
        missing = [field for field in required if field not in fieldnames]
        if missing:
            raise ValueError(f"metadata CSV is missing columns: {missing}")
        metadata_rows = list(reader)
    if not metadata_rows:
        raise ValueError(f"metadata CSV is empty: {metadata_csv}")

    spectra: list[np.ndarray] = []
    source_ids: list[str] = []
    source_files: list[str] = []
    smiles: list[str] = []
    phases: list[str] = []
    rejected: list[dict[str, object]] = []
    for row_number, row in enumerate(metadata_rows, start=2):
        source_id = (row.get(id_field) or "").strip() or f"row-{row_number}"
        try:
            relative_image = _clean_required(row, image_field, row_number)
            image_path = Path(relative_image).expanduser()
            if not image_path.is_absolute():
                image_path = root / image_path
            row_smiles = (
                _clean_required(row, smiles_field, row_number) if smiles_field else None
            )
            row_phase = None
            if phase_field:
                raw_phase = _clean_required(row, phase_field, row_number)
                row_phase = normalize_phase(raw_phase)
                if row_phase is None:
                    raise ValueError(f"CSV row {row_number}: unsupported phase {raw_phase!r}")
            elif detect_phase_with_ocr:
                row_phase = detect_phase(image_path)
                if row_phase is None:
                    raise ValueError(f"CSV row {row_number}: phase OCR produced no supported phase")
            spectrum = extract_spectrum(
                image_path,
                target_points=target_points,
                low=low,
                high=high,
                threshold=threshold,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            if not skip_invalid:
                raise
            rejected.append(
                {"row": row_number, "source_id": source_id, "error": str(exc)}
            )
            continue
        spectra.append(spectrum)
        source_ids.append(source_id)
        source_files.append(relative_image)
        if row_smiles is not None:
            smiles.append(row_smiles)
        if row_phase is not None:
            phases.append(row_phase)

    if not spectra:
        raise RuntimeError(f"no valid SDBS images referenced by {metadata_csv}")

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "ir.npy", np.stack(spectra).astype(np.float32, copy=False))
    np.save(
        output_dir / "wavenumbers.npy",
        np.linspace(low, high, target_points, dtype=np.float32),
    )
    np.save(output_dir / "source_ids.npy", np.asarray(source_ids, dtype=str))
    np.save(output_dir / "source_files.npy", np.asarray(source_files, dtype=str))
    if smiles_field:
        np.save(output_dir / "smiles.npy", np.asarray(smiles, dtype=str))
    if phase_field or detect_phase_with_ocr:
        np.save(output_dir / "phase.npy", np.asarray(phases, dtype=str))

    manifest = {
        "source": "SDBS PNG spectrum plots",
        "metadata_csv": str(metadata_csv),
        "rows": len(spectra),
        "rejected_rows": len(rejected),
        "signal_length": target_points,
        "range_cm-1": [low, high],
        "axis_order": "ascending",
        "smiles_aligned": smiles_field is not None,
        "phase_aligned": phase_field is not None or detect_phase_with_ocr,
        "rejected": rejected,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"saved {len(spectra)} SDBS spectra to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--target-points", type=int, default=3600)
    parser.add_argument("--low", type=float, default=400.0)
    parser.add_argument("--high", type=float, default=4000.0)
    parser.add_argument("--threshold", type=int, default=160)
    parser.add_argument("--image-field", default="Image File")
    parser.add_argument("--id-field", default="SDBS No")
    parser.add_argument("--smiles-field")
    parser.add_argument("--phase-field")
    parser.add_argument("--detect-phase", action="store_true")
    parser.add_argument("--skip-invalid", action="store_true")
    args = parser.parse_args()
    prepare(
        args.metadata_csv,
        args.output_dir,
        image_root=args.image_root,
        target_points=args.target_points,
        low=args.low,
        high=args.high,
        threshold=args.threshold,
        image_field=args.image_field,
        id_field=args.id_field,
        smiles_field=args.smiles_field,
        phase_field=args.phase_field,
        detect_phase_with_ocr=args.detect_phase,
        skip_invalid=args.skip_invalid,
    )


if __name__ == "__main__":
    main()
