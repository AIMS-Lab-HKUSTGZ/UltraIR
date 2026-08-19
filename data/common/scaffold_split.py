"""Scaffold-aware splitting for aligned molecular and mixture arrays."""

from __future__ import annotations

import argparse
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np

from .aligned_arrays import DEFAULT_STATIC_ARRAYS

INVALID = "__INVALID__"
NO_SCAFFOLD = "__NO_SCAFFOLD__"


@lru_cache(maxsize=100_000)
def scaffold_key(smiles: object) -> str:
    try:
        from rdkit import Chem, rdBase
        from rdkit.Chem.Scaffolds import MurckoScaffold
    except ImportError as exc:
        raise RuntimeError("scaffold splitting requires RDKit") from exc
    text = smiles.decode("utf-8", errors="strict") if isinstance(smiles, bytes) else str(smiles)
    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(text)
    if mol is None:
        return INVALID
    core = MurckoScaffold.GetScaffoldForMol(mol)
    if core is None or core.GetNumAtoms() == 0:
        return NO_SCAFFOLD
    return Chem.MolToSmiles(core, isomericSmiles=False)


def _ordered_group_ids(
    group_ids: list[int], groups: list[list[int]], rng: np.random.Generator
) -> list[int]:
    ordered = list(group_ids)
    rng.shuffle(ordered)
    ordered.sort(key=lambda group_id: -len(groups[group_id]))
    return ordered


def _validation_groups(
    candidates: list[int],
    groups: list[list[int]],
    target: int,
    rng: np.random.Generator,
) -> set[int]:
    """Choose intact groups whose total size is close to the requested target."""
    ordered = _ordered_group_ids(candidates, groups, rng)
    chosen: set[int] = set()
    size = 0
    for group_id in ordered:
        proposed = size + len(groups[group_id])
        if abs(target - proposed) < abs(target - size):
            chosen.add(group_id)
            size = proposed
    if target > 0 and not chosen and candidates:
        chosen.add(min(candidates, key=lambda item: abs(len(groups[item]) - target)))
    return chosen


def scaffold_split(
    smiles: np.ndarray,
    k: int = 5,
    seed: int = 42,
    valid_fraction: float = 0.1,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Build deterministic scaffold-separated train/valid/test index triples.

    Test folds contain approximately ``1 / k`` of all rows. Validation is
    selected independently from the remaining scaffold groups and targets
    ``valid_fraction`` of the complete dataset, giving approximately 70/10/20
    for the default five-fold configuration.
    """
    values = np.asarray(smiles)
    if values.ndim != 1 or k < 2:
        raise ValueError("smiles must be 1D and k must be >= 2")
    if len(values) < k:
        raise ValueError(f"need at least k={k} molecules, got {len(values)}")
    if not 0.0 < valid_fraction < 1.0 - (1.0 / k):
        raise ValueError(
            f"valid_fraction must be between 0 and {1.0 - (1.0 / k):.3f}"
        )
    groups: dict[str, list[int]] = defaultdict(list)
    for i, value in enumerate(values):
        groups[scaffold_key(value)].append(i)
    rng = np.random.default_rng(seed)
    grouped_rows = [
        list(ids)
        for key, ids in groups.items()
        if key not in (INVALID, NO_SCAFFOLD)
    ]
    grouped_rows.extend(
        [row]
        for key in (INVALID, NO_SCAFFOLD)
        for row in groups.get(key, [])
    )
    all_group_ids = list(range(len(grouped_rows)))
    ordered = _ordered_group_ids(all_group_ids, grouped_rows, rng)
    buckets: list[list[int]] = [[] for _ in range(k)]
    sizes = np.zeros(k, dtype=np.int64)
    for group_id in ordered:
        bucket = int(np.argmin(sizes))
        buckets[bucket].append(group_id)
        sizes[bucket] += len(grouped_rows[group_id])

    result = []
    for fold in range(k):
        test_groups = set(buckets[fold])
        candidates = [group_id for group_id in all_group_ids if group_id not in test_groups]
        valid_groups = _validation_groups(
            candidates,
            grouped_rows,
            target=int(round(len(values) * valid_fraction)),
            rng=np.random.default_rng(seed + 10_000 + fold),
        )
        train_groups = set(candidates) - valid_groups

        def rows(group_ids: set[int]) -> np.ndarray:
            return np.asarray(
                sorted(row for group_id in group_ids for row in grouped_rows[group_id]),
                dtype=np.int64,
            )

        train = rows(train_groups)
        valid = rows(valid_groups)
        test = rows(test_groups)
        combined = np.concatenate([train, valid, test])
        if len(combined) != len(values) or len(np.unique(combined)) != len(values):
            raise RuntimeError(f"fold {fold + 1} is not a complete disjoint partition")
        result.append((train, valid, test))
    return result


def validate_mixture_references(
    references: np.ndarray, n_molecules: int, mixture_per_ref: int = 0
) -> np.ndarray:
    values = np.asarray(references)
    if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
        raise ValueError("mixture_ref_idx.npy must be a 1D integer array")
    values = values.astype(np.int64, copy=False)
    if values.size == 0:
        raise ValueError("mixture_ref_idx.npy is empty")
    if values.min() < 0 or values.max() >= n_molecules:
        raise ValueError(
            f"mixture references must be in [0, {n_molecules}), got "
            f"[{int(values.min())}, {int(values.max())}]"
        )
    if mixture_per_ref:
        counts = np.bincount(values, minlength=n_molecules)
        bad = np.flatnonzero(counts != mixture_per_ref)
        if bad.size:
            first = int(bad[0])
            raise ValueError(
                f"reference {first} has {int(counts[first])} rows; "
                f"expected {mixture_per_ref}"
            )
    return values


def _save_split(
    input_dir: Path,
    output_dir: Path,
    indices: np.ndarray,
    n_molecules: int,
    mixture_references: np.ndarray | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    mixture_rows = (
        np.flatnonzero(np.isin(mixture_references, indices))
        if mixture_references is not None
        else np.empty((0,), dtype=np.int64)
    )
    for path in sorted(input_dir.glob("*.npy")):
        values = np.load(path, allow_pickle=True)
        if values.ndim == 0:
            raise ValueError(f"{path.name} is scalar")
        is_mixture = path.name.startswith("mixture_")
        if path.name in DEFAULT_STATIC_ARRAYS:
            selected = values
        elif is_mixture:
            if mixture_references is None:
                raise ValueError(
                    f"{path.name} requires mixture_ref_idx.npy in {input_dir}"
                )
            if values.shape[0] != len(mixture_references):
                raise ValueError(
                    f"{path.name}: first dimension {values.shape[0]} does not match "
                    f"mixture references {len(mixture_references)}"
                )
            selected = values[mixture_rows]
        elif values.shape[0] == n_molecules:
            selected = values[indices]
        else:
            raise ValueError(
                f"{path.name}: first dimension {values.shape[0]} does not match "
                f"molecule count {n_molecules}"
            )
        np.save(output_dir / path.name, selected)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create scaffold-aware fold-* directories from aligned .npy files."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smiles", default="smiles.npy")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--valid-fraction", type=float, default=0.1)
    parser.add_argument("--mixture-per-ref", type=int, default=0)
    args = parser.parse_args()
    smiles = np.load(args.input_dir / args.smiles, allow_pickle=True)
    reference_path = args.input_dir / "mixture_ref_idx.npy"
    mixture_references = None
    if reference_path.exists():
        mixture_references = validate_mixture_references(
            np.load(reference_path, allow_pickle=False),
            len(smiles),
            args.mixture_per_ref,
        )
    elif args.mixture_per_ref:
        raise FileNotFoundError(reference_path)
    folds = scaffold_split(
        smiles,
        k=args.k,
        seed=args.seed,
        valid_fraction=args.valid_fraction,
    )
    for fold, (train, valid, test) in enumerate(folds, 1):
        for name, indices in (("train", train), ("valid", valid), ("test", test)):
            _save_split(args.input_dir, args.output_dir / f"fold-{fold}" / name,
                        indices, len(smiles), mixture_references)
        print(f"fold-{fold}: train={len(train)} valid={len(valid)} test={len(test)}")


if __name__ == "__main__":
    main()
