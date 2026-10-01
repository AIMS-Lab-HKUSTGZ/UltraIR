"""Generate grouped positive/negative mixture pairs for targeted tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .spectra import minmax_normalize


def generate_pairs(
    pure_library: np.ndarray,
    augmentations: int = 4,
    seed: int = 42,
    normalize_components: bool = False,
    normalize_mixtures: bool = False,
    outputs: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    values = np.asarray(pure_library, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] < 6:
        raise ValueError("pure_library must be [N, L] with at least six spectra")
    if values.shape[1] == 0 or not np.isfinite(values).all():
        raise ValueError("pure_library must contain finite, non-empty spectra")
    if augmentations < 1:
        raise ValueError("augmentations must be positive")
    rng = np.random.default_rng(seed)
    n, length = values.shape
    components = (
        np.stack([minmax_normalize(row) for row in values], axis=0)
        if normalize_components
        else values
    )
    if not normalize_components and (
        float(components.min()) < -1e-6 or float(components.max()) > 1.0 + 1e-6
    ):
        raise ValueError(
            "pure_library must be in [0, 1] unless --normalize-components is used"
        )
    if outputs is None:
        rows = n * augmentations * 2
        outputs = {
            "mixture_set": np.empty((rows, 2, length), dtype=np.float32),
            "mixture_labels": np.empty(rows, dtype=np.int64),
            "mixture_ref_idx": np.empty(rows, dtype=np.int64),
            "mixture_ref_weight": np.empty(rows, dtype=np.float32),
        }
    cursor = 0
    def store(ref, mixture, label, weight):
        nonlocal cursor
        outputs["mixture_set"][cursor, 0] = components[ref]
        outputs["mixture_set"][cursor, 1] = mixture
        outputs["mixture_labels"][cursor] = label
        outputs["mixture_ref_idx"][cursor] = ref
        outputs["mixture_ref_weight"][cursor] = weight
        cursor += 1
    all_indices = np.arange(n, dtype=np.int64)
    for ref in all_indices:
        for _ in range(augmentations):
            count = int(rng.integers(1, min(4, n - 1) + 1))
            others = rng.choice(n - 1, size=count, replace=False)
            others += others >= ref
            coef_ref = float(rng.uniform(0.1, 0.9))
            coeffs = rng.uniform(0.05, 0.5, size=count).astype(np.float32)
            total = coef_ref + float(coeffs.sum())
            coef_ref, coeffs = coef_ref / total, coeffs / total
            mixture = coef_ref * components[ref] + np.sum(
                coeffs[:, None] * components[others], axis=0
            )
            mixture += rng.normal(0.0, 0.005, size=length).astype(np.float32)
            mixture += np.linspace(
                rng.uniform(-0.02, 0.02),
                rng.uniform(-0.02, 0.02),
                length,
                dtype=np.float32,
            )
            if normalize_mixtures:
                mixture = minmax_normalize(mixture)
            else:
                mixture = np.clip(mixture, 0.0, 1.0)
            store(ref, mixture, 1, coef_ref)
            neg_count = int(rng.integers(2, min(5, n - 1) + 1))
            neg = rng.choice(n - 1, size=neg_count, replace=False)
            neg += neg >= ref
            neg_coeffs = rng.uniform(0.1, 0.8, size=neg_count).astype(np.float32)
            neg_coeffs /= float(neg_coeffs.sum())
            negative = np.sum(neg_coeffs[:, None] * components[neg], axis=0)
            negative += rng.normal(0.0, 0.005, size=length).astype(np.float32)
            negative += np.linspace(
                rng.uniform(-0.02, 0.02),
                rng.uniform(-0.02, 0.02),
                length,
                dtype=np.float32,
            )
            negative = (
                minmax_normalize(negative)
                if normalize_mixtures
                else np.clip(negative, 0.0, 1.0)
            )
            store(ref, negative, 0, 0.0)
    return outputs


def generate_pairs_to_disk(pure_library, output_dir: Path, augmentations=4, seed=42):
    """Write pairs with bounded memory; call independently on each scaffold split."""
    from contextlib import ExitStack
    from .array_writer import array_writer
    output_dir = Path(output_dir)
    rows = len(pure_library) * augmentations * 2
    shapes = {
        "mixture_set": ((rows, 2, pure_library.shape[-1]), np.float32),
        "mixture_labels": ((rows,), np.int64),
        "mixture_ref_idx": ((rows,), np.int64),
        "mixture_ref_weight": ((rows,), np.float32),
    }
    with ExitStack() as stack:
        outputs = {name: stack.enter_context(array_writer(output_dir / (name + ".npy"), shape, dtype))
                   for name, (shape, dtype) in shapes.items()}
        generate_pairs(pure_library, augmentations=augmentations, seed=seed, outputs=outputs)
    (output_dir / "pairs_complete.json").write_text(json.dumps({
        "molecules": len(pure_library), "pairs": rows,
        "augmentations": augmentations, "seed": seed,
        "component_pool": "this split only",
    }, indent=2) + "\n")


def save_pairs(
    outputs: dict[str, np.ndarray],
    output_dir: Path,
    smiles: np.ndarray | None = None,
    k: int = 5,
    seed: int = 42,
    valid_fraction: float = 0.1,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    row_count = len(outputs["mixture_ref_idx"])
    if any(len(values) != row_count for values in outputs.values()):
        raise ValueError("pair output arrays are not row-aligned")
    for name, values in outputs.items():
        np.save(output_dir / f"{name}.npy", values)

    manifest: dict[str, object] = {"pairs": row_count, "folds": 0}
    if smiles is not None:
        from .scaffold_split import scaffold_split

        values = np.asarray(smiles)
        references = outputs["mixture_ref_idx"]
        if values.ndim != 1:
            raise ValueError(f"smiles must be 1D, got {values.shape}")
        if references.min() < 0 or references.max() >= len(values):
            raise ValueError("mixture_ref_idx contains rows outside the SMILES array")
        folds = scaffold_split(
            values,
            k=k,
            seed=seed,
            valid_fraction=valid_fraction,
        )
        for fold, (train, valid, test) in enumerate(folds, 1):
            for split, molecule_rows in (
                ("train", train),
                ("valid", valid),
                ("test", test),
            ):
                pair_rows = np.flatnonzero(np.isin(references, molecule_rows))
                split_dir = output_dir / f"fold-{fold}" / split
                split_dir.mkdir(parents=True, exist_ok=True)
                for name, array in outputs.items():
                    np.save(split_dir / f"{name}.npy", array[pair_rows])
        manifest.update(
            {
                "folds": k,
                "fold_method": "scaffold",
                "valid_fraction": valid_fraction,
            }
        )
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate mixture_set/labels/ref_idx arrays.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--augmentations", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--normalize-components", action="store_true")
    parser.add_argument("--normalize-mixtures", action="store_true")
    parser.add_argument(
        "--smiles",
        type=Path,
        help="Aligned smiles.npy; when provided, also create scaffold folds",
    )
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--valid-fraction", type=float, default=0.1)
    args = parser.parse_args()
    library = np.load(args.input, allow_pickle=False)
    outputs = generate_pairs(
        library,
        args.augmentations,
        args.seed,
        args.normalize_components,
        args.normalize_mixtures,
    )
    smiles = np.load(args.smiles, allow_pickle=True) if args.smiles else None
    if smiles is not None and len(smiles) != len(library):
        raise ValueError(
            f"IR/SMILES alignment mismatch: {len(library)} vs {len(smiles)}"
        )
    save_pairs(
        outputs,
        args.output_dir,
        smiles,
        args.k,
        args.seed,
        args.valid_fraction,
    )
    print(f"saved {len(outputs['mixture_labels'])} pairs to {args.output_dir}")


if __name__ == "__main__":
    main()
