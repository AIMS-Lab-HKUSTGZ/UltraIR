"""Dispatch and validate data preparation from a YAML recipe."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


PROCESSORS = {
    "aligned_filter",
    "aligned_subset",
    "ftir_mix",
    "jcamp",
    "mixture_csv",
    "molecular",
    "pairs",
    "parquet_ir",
    "sdbs_image",
}


def _section(config: dict[str, Any]) -> dict[str, Any]:
    value = config.get("prepare")
    if not isinstance(value, dict):
        raise ValueError("config.prepare must be a mapping")
    processor = value.get("processor")
    if processor not in PROCESSORS:
        raise ValueError(f"prepare.processor must be one of: {', '.join(sorted(PROCESSORS))}")
    return value


def _options(section: dict[str, Any]) -> dict[str, Any]:
    value = section.get("options", {})
    if not isinstance(value, dict):
        raise ValueError("prepare.options must be a mapping")
    return value


def _input_output(section: dict[str, Any]) -> tuple[Path, Path]:
    if not section.get("input") or not section.get("output"):
        raise ValueError("prepare.input and prepare.output are required")
    return Path(section["input"]).expanduser(), Path(section["output"]).expanduser()


def _empty_output(output: Path, overwrite: bool) -> None:
    if output.exists() and not output.is_dir():
        raise NotADirectoryError(output)
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError(
            f"output directory is not empty: {output}; "
            "set prepare.overwrite=true to replace owned files"
        )


def prepare_from_config(config: dict[str, Any]) -> dict[str, Any]:
    section = _section(config)
    options = _options(section)
    input_path, output_path = _input_output(section)
    _empty_output(output_path, bool(section.get("overwrite", False)))
    processor = str(section["processor"])
    seed = int(options.get("seed", config.get("seed", 42)))

    if processor == "molecular":
        from .molecular import prepare

        prepare(
            input_path / str(options.get("ir_name", "ir.npy")),
            input_path / str(options.get("smiles_name", "smiles.npy")),
            output_path,
            normalize=bool(options.get("normalize", True)),
            make_folds=bool(options.get("make_folds", True)),
            k=int(options.get("k", 5)),
            seed=seed,
            keep_invalid=bool(options.get("keep_invalid", False)),
            valid_fraction=float(options.get("valid_fraction", 0.1)),
            target_points=int(options.get("target_points", 0)),
        )
    elif processor == "jcamp":
        from .jcamp import prepare

        smiles_dir = options.get("smiles_dir")
        prepare(
            input_path,
            output_path,
            target_points=int(options.get("target_points", 3600)),
            low=float(options.get("low", 400.0)),
            high=float(options.get("high", 4000.0)),
            smiles_dir=Path(smiles_dir).expanduser() if smiles_dir else None,
            skip_invalid=bool(options.get("skip_invalid", False)),
        )
    elif processor == "parquet_ir":
        from .parquet_ir import prepare

        prepare(
            input_path,
            output_path,
            pattern=str(options.get("pattern", "*.parquet")),
            target_points=int(options.get("target_points", 3600)),
            low=float(options.get("low", 400.0)),
            high=float(options.get("high", 4000.0)),
            batch_size=int(options.get("batch_size", 512)),
        )
    elif processor == "sdbs_image":
        from .sdbs_image import prepare

        image_root = options.get("image_root")
        prepare(
            input_path,
            output_path,
            image_root=Path(image_root).expanduser() if image_root else None,
            target_points=int(options.get("target_points", 3600)),
            low=float(options.get("low", 400.0)),
            high=float(options.get("high", 4000.0)),
            threshold=int(options.get("threshold", 160)),
            image_field=str(options.get("image_field", "Image File")),
            id_field=str(options.get("id_field", "SDBS No")),
            smiles_field=options.get("smiles_field"),
            phase_field=options.get("phase_field"),
            detect_phase_with_ocr=bool(options.get("detect_phase", False)),
            skip_invalid=bool(options.get("skip_invalid", False)),
        )
    elif processor == "mixture_csv":
        from .mixture_csv import prepare

        prepare(
            input_path / str(options.get("spectra_name", "spectra.csv")),
            input_path / str(options.get("targets_name", "targets.csv")),
            output_path,
            target_points=int(options.get("target_points", 0)),
            normalize=bool(options.get("normalize", True)),
        )
    elif processor == "pairs":
        from .pairs import generate_pairs, save_pairs

        library = np.load(
            input_path / str(options.get("ir_name", "ir_norm.npy")),
            allow_pickle=False,
        )
        outputs = generate_pairs(
            library,
            augmentations=int(options.get("augmentations", 4)),
            seed=seed,
            normalize_components=bool(options.get("normalize_components", False)),
            normalize_mixtures=bool(options.get("normalize_mixtures", False)),
        )
        smiles_name = options.get("smiles_name", "smiles.npy")
        smiles_path = input_path / str(smiles_name)
        smiles = np.load(smiles_path, allow_pickle=True) if smiles_path.is_file() else None
        if smiles is not None and len(smiles) != len(library):
            raise ValueError(
                f"IR/SMILES alignment mismatch: {len(library)} vs {len(smiles)}"
            )
        save_pairs(
            outputs,
            output_path,
            smiles,
            k=int(options.get("k", 5)),
            seed=seed,
            valid_fraction=float(options.get("valid_fraction", 0.1)),
        )
    elif processor == "aligned_filter":
        from .aligned_arrays import filter_directory

        filter_directory(
            input_path,
            output_path,
            mask_name=str(options.get("mask", "valid_mask.npy")),
            reference_name=str(options.get("reference", "smiles.npy")),
        )
    elif processor == "aligned_subset":
        from .aligned_arrays import subset_directory

        subset_directory(
            input_path,
            output_path,
            ratio=float(options["ratio"]),
            seed=seed,
            reference_name=str(options.get("reference", "smiles.npy")),
        )
    else:
        from data.mixture_level_component_quantification.ftir_mix import prepare

        prepare(input_path, output_path, seed=seed)
    return validate_from_config(config)


def _aligned_rows(paths: list[Path]) -> int:
    arrays = []
    for path in paths:
        try:
            arrays.append(np.load(path, mmap_mode="r", allow_pickle=True))
        except ValueError:
            arrays.append(np.load(path, allow_pickle=True))
    if any(array.ndim == 0 for array in arrays):
        raise ValueError(f"scalar array among {[str(path) for path in paths]}")
    sizes = {int(array.shape[0]) for array in arrays}
    if len(sizes) != 1:
        raise ValueError(
            f"unaligned arrays: {[(path.name, array.shape) for path, array in zip(paths, arrays)]}"
        )
    return sizes.pop()


def validate_from_config(config: dict[str, Any]) -> dict[str, Any]:
    section = _section(config)
    output = Path(section["output"]).expanduser()
    processor = str(section["processor"])
    if not output.is_dir():
        raise FileNotFoundError(output)

    groups: list[list[Path]] = []
    if processor == "molecular":
        groups.append([output / "ir_norm.npy", output / "smiles.npy", output / "valid_mask.npy"])
    elif processor in {"jcamp", "parquet_ir", "sdbs_image"}:
        names = (
            ["ir_norm.npy", "source_ids.npy"]
            if processor == "jcamp"
            else (
                ["ir_norm.npy", "ids.npy", "smiles.npy"]
                if processor == "parquet_ir"
                else ["ir.npy", "source_ids.npy", "source_files.npy"]
            )
        )
        groups.append([output / name for name in names])
    elif processor == "mixture_csv":
        groups.append([output / "spectra.npy", output / "targets.npy"])
    elif processor == "pairs":
        groups.append(
            [
                output / "mixture_set.npy",
                output / "mixture_labels.npy",
                output / "mixture_ref_idx.npy",
                output / "mixture_ref_weight.npy",
            ]
        )
    elif processor == "ftir_mix":
        for fold in range(1, 6):
            for split in ("train", "valid", "test"):
                root = output / f"fold-{fold}" / split
                groups.extend(
                    [
                        [
                            root / "experimental_four_component_spectra.npy",
                            root / "experimental_four_component_targets.npy",
                        ],
                        [
                            root / "synthetic_four_component_spectra.npy",
                            root / "synthetic_four_component_targets.npy",
                        ],
                    ]
                )
    else:
        reference = str(_options(section).get("reference", "smiles.npy"))
        groups.append([output / reference, output / "source_indices.npy"])

    rows = []
    for paths in groups:
        missing = [path for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(", ".join(str(path) for path in missing))
        rows.append(_aligned_rows(paths))
    return {
        "processor": processor,
        "output": str(output.resolve()),
        "validated_groups": len(groups),
        "rows_per_group": rows,
    }
