"""Prepare the configured four-component FTIRMix2022 datasets."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from data.common.splits import train_valid_test_folds


EXPERIMENTAL_COLUMNS = ("F AN", "F ADN", "F PN", "F glycerol")
EXPERIMENTAL_CONSTANTS = np.asarray([4.1, 4.2, 3.8, 5.0], dtype=np.float64) / 100.0
SYNTHETIC_COMPONENTS = ("Acrylonitrile", "Adiponitrile", "Propionitrile", "EDTA")


@dataclass(frozen=True)
class MixtureBundle:
    spectra: np.ndarray
    targets: np.ndarray
    wavenumbers: np.ndarray
    sample_ids: np.ndarray
    groups: np.ndarray
    component_names: tuple[str, ...]


def _pandas():
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("FTIRMix preparation requires pandas") from exc
    return pd


def _read_numeric(path: Path):
    pd = _pandas()
    frame = pd.read_csv(path, header=None).apply(pd.to_numeric, errors="coerce")
    frame = frame.dropna(axis=0, how="all").dropna(axis=1, how="all")
    if frame.empty:
        raise ValueError(f"no numeric content in {path}")
    return frame.reset_index(drop=True)


def _read_sample_spectrum(path: Path) -> tuple[np.ndarray, np.ndarray]:
    frame = _read_numeric(path)
    if frame.shape[1] < 2:
        raise ValueError(f"expected two columns in {path}")
    axis = frame.iloc[:, 0].to_numpy(dtype=np.float64)
    signal = frame.iloc[:, 1].to_numpy(dtype=np.float64)
    finite = np.isfinite(axis) & np.isfinite(signal)
    axis, signal = axis[finite], signal[finite]
    if len(axis) < 2:
        raise ValueError(f"fewer than two finite spectrum points in {path}")
    order = np.argsort(axis, kind="stable")
    return axis[order].astype(np.float32), signal[order].astype(np.float32)


def _read_component_spectrum(path: Path) -> tuple[np.ndarray, np.ndarray]:
    frame = _read_numeric(path)
    if frame.shape[0] < 2:
        raise ValueError(f"expected wavenumber and intensity rows in {path}")
    axis = frame.iloc[0].to_numpy(dtype=np.float64)
    signal = frame.iloc[1].to_numpy(dtype=np.float64)
    finite = np.isfinite(axis) & np.isfinite(signal)
    axis, signal = axis[finite], signal[finite]
    if len(axis) < 2:
        raise ValueError(f"fewer than two finite spectrum points in {path}")
    order = np.argsort(axis, kind="stable")
    return axis[order].astype(np.float32), signal[order].astype(np.float32)


def _stack_records(
    records: list[tuple[str, str, np.ndarray, np.ndarray, np.ndarray]],
    component_names: tuple[str, ...],
) -> MixtureBundle:
    if not records:
        raise ValueError("no FTIRMix records found")
    reference_axis = records[0][2]
    for sample_id, _group, axis, signal, target in records:
        if axis.shape != reference_axis.shape or not np.allclose(
            axis, reference_axis, atol=1e-4, rtol=0.0
        ):
            raise ValueError(f"wavenumber grid mismatch for {sample_id}")
        if len(signal) != len(reference_axis) or not np.isfinite(signal).all():
            raise ValueError(f"invalid signal for {sample_id}")
        if target.shape != (len(component_names),) or not np.isfinite(target).all():
            raise ValueError(f"invalid target for {sample_id}")
    return MixtureBundle(
        spectra=np.stack([record[3] for record in records]).astype(np.float32),
        targets=np.stack([record[4] for record in records]).astype(np.float32),
        wavenumbers=reference_axis.astype(np.float32),
        sample_ids=np.asarray([record[0] for record in records], dtype=str),
        groups=np.asarray([record[1] for record in records], dtype=str),
        component_names=component_names,
    )


def load_experimental(source_root: Path) -> MixtureBundle:
    pd = _pandas()
    subset_root = source_root / "Experimental data" / "4-AN"
    if not subset_root.is_dir():
        raise FileNotFoundError(subset_root)
    records = []
    for run_dir in sorted(path for path in subset_root.iterdir() if path.is_dir()):
        labels_path = run_dir / "Labels.csv"
        if not labels_path.is_file():
            continue
        labels = pd.read_csv(labels_path)
        if "Sample" not in labels.columns:
            raise KeyError(f"{labels_path} is missing Sample")
        missing = [name for name in EXPERIMENTAL_COLUMNS if name not in labels.columns]
        if missing:
            raise KeyError(f"{labels_path} is missing columns {missing}")
        labels = labels.set_index("Sample")
        lookup = {str(value).casefold(): value for value in labels.index}
        for sample_path in sorted(
            path
            for path in run_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() == ".csv"
            and path.name.casefold() != "labels.csv"
        ):
            label_key = lookup.get(sample_path.name.casefold())
            if label_key is None:
                raise KeyError(f"{sample_path.name} has no matching row in {labels_path}")
            raw = labels.loc[label_key, list(EXPERIMENTAL_COLUMNS)].to_numpy(dtype=np.float64)
            denominator = float(raw.sum())
            if not np.isfinite(raw).all() or denominator <= 0:
                raise ValueError(f"invalid composition for {sample_path}")
            target = raw * EXPERIMENTAL_CONSTANTS / denominator
            axis, signal = _read_sample_spectrum(sample_path)
            records.append(
                (f"{run_dir.name}:{sample_path.name}", run_dir.name, axis, signal, target)
            )
    return _stack_records(records, ("an_pct", "adn_pct", "pn_pct", "glycerol_pct"))


def _load_components(molecule_dir: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    records = []
    for path in sorted(
        item
        for item in molecule_dir.iterdir()
        if item.is_file() and item.suffix.lower() == ".csv"
    ):
        axis, signal = _read_component_spectrum(path)
        records.append((path.stem, axis, signal))
    if not records:
        raise FileNotFoundError(f"no component CSV files under {molecule_dir}")
    spans = [float(axis[-1] - axis[0]) for _, axis, _ in records]
    reference_axis = records[int(np.argmin(spans))][1]
    spectra = [
        np.interp(reference_axis, axis, signal).astype(np.float32)
        for _, axis, signal in records
    ]
    return [name for name, _, _ in records], reference_axis, np.stack(spectra)


def load_synthetic(source_root: Path, seed: int = 42) -> MixtureBundle:
    try:
        from scipy.stats import qmc
    except ImportError as exc:
        raise RuntimeError("synthetic FTIRMix preparation requires scipy") from exc
    molecule_dir = (
        source_root
        / "Synthetic data"
        / "4 components, AN ADN PN EDTA"
        / "Molecules"
    )
    names, axis, pure = _load_components(molecule_dir)
    lookup = {name.casefold(): index for index, name in enumerate(names)}
    missing = [name for name in (*SYNTHETIC_COMPONENTS, "Water") if name.casefold() not in lookup]
    if missing:
        raise KeyError(f"missing component spectra: {missing}; available={names}")
    target_indices = [lookup[name.casefold()] for name in SYNTHETIC_COMPONENTS]
    water_index = lookup["water"]

    rng = np.random.default_rng(seed)
    sample_count = 400
    random_compositions = rng.uniform(0.0, 0.1, size=(sample_count, 4)).astype(np.float32)
    sobol = qmc.Sobol(d=4, scramble=False, seed=seed)
    sobol_compositions = sobol.random_base2(m=9).astype(np.float32)[:sample_count] * 0.1
    records = []

    def mixture(coefficients: np.ndarray) -> np.ndarray:
        weights = np.zeros(len(names), dtype=np.float32)
        weights[target_indices] = coefficients
        weights[water_index] = 1.0 - float(coefficients.sum())
        return (weights @ pure).astype(np.float32)

    for index, coefficients in enumerate(sobol_compositions):
        records.append(
            (
                f"train_sobol_{index:04d}",
                "train_sobol",
                axis,
                mixture(coefficients),
                coefficients,
            )
        )
    for noise_factor in np.linspace(0.0, 1.0, 5, dtype=np.float32):
        for index, coefficients in enumerate(random_compositions):
            signal = mixture(coefficients)
            peak_region = signal[1000:] if len(signal) > 1000 else signal
            amplitude = float(peak_region.max()) * float(noise_factor)
            noise = rng.uniform(-0.05, 0.05, size=len(signal)).astype(np.float32)
            signal = signal + noise * amplitude
            group = f"test_nf_{float(noise_factor):.2f}"
            records.append(
                (
                    f"test_rand_nf{float(noise_factor):.2f}_{index:04d}",
                    group,
                    axis,
                    signal,
                    coefficients,
                )
            )
    return _stack_records(records, ("an_pct", "adn_pct", "pn_pct", "edta_pct"))


def _save_bundle(root: Path, prefix: str, bundle: MixtureBundle) -> None:
    root.mkdir(parents=True, exist_ok=True)
    np.save(root / f"{prefix}_spectra.npy", bundle.spectra)
    np.save(root / f"{prefix}_targets.npy", bundle.targets)
    np.save(root / f"{prefix}_wavenumbers.npy", bundle.wavenumbers)
    np.save(root / f"{prefix}_sample_ids.npy", bundle.sample_ids)
    np.save(root / f"{prefix}_groups.npy", bundle.groups)
    np.save(root / f"{prefix}_component_names.npy", np.asarray(bundle.component_names, dtype=str))


def prepare(source_root: Path, output_dir: Path, seed: int = 42) -> None:
    experimental = load_experimental(source_root)
    synthetic = load_synthetic(source_root, seed)
    _save_bundle(output_dir / "full_data", "experimental_four_component", experimental)
    _save_bundle(output_dir / "full_data", "synthetic_four_component", synthetic)
    experimental_folds = train_valid_test_folds(len(experimental.spectra), seed=seed)
    synthetic_folds = train_valid_test_folds(
        len(synthetic.spectra), seed=seed, stratify=synthetic.groups
    )
    for fold, (experimental_indices, synthetic_indices) in enumerate(
        zip(experimental_folds, synthetic_folds), start=1
    ):
        for split_index, split in enumerate(("train", "valid", "test")):
            split_dir = output_dir / f"fold-{fold}" / split
            split_dir.mkdir(parents=True, exist_ok=True)
            exp_rows = experimental_indices[split_index]
            syn_rows = synthetic_indices[split_index]
            np.save(
                split_dir / "experimental_four_component_spectra.npy",
                experimental.spectra[exp_rows],
            )
            np.save(
                split_dir / "experimental_four_component_targets.npy",
                experimental.targets[exp_rows],
            )
            np.save(
                split_dir / "synthetic_four_component_spectra.npy",
                synthetic.spectra[syn_rows],
            )
            np.save(
                split_dir / "synthetic_four_component_targets.npy",
                synthetic.targets[syn_rows],
            )
    manifest = {
        "source": str(source_root.resolve()),
        "seed": seed,
        "fold_ratio": {"train": 0.7, "valid": 0.1, "test": 0.2},
        "experimental_rows": len(experimental.spectra),
        "synthetic_rows": len(synthetic.spectra),
        "experimental_signal_length": experimental.spectra.shape[1],
        "synthetic_signal_length": synthetic.spectra.shape[1],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prepare(args.source_root, args.output_dir, args.seed)


if __name__ == "__main__":
    main()
