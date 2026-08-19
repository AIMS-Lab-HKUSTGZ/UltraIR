"""Reproducible train/validation/test folds for non-molecular datasets."""

from __future__ import annotations

import numpy as np


def train_valid_test_folds(
    n_rows: int,
    k: int = 5,
    seed: int = 42,
    valid_fraction: float = 0.1,
    stratify: np.ndarray | None = None,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Return k disjoint folds with approximately 70/10/20 default ratios."""
    if k < 2 or n_rows < k:
        raise ValueError(f"need n_rows >= k >= 2, got n_rows={n_rows}, k={k}")
    if not 0.0 < valid_fraction < 1.0 - (1.0 / k):
        raise ValueError("valid_fraction leaves no room for train/test rows")
    rng = np.random.default_rng(seed)
    buckets: list[list[int]] = [[] for _ in range(k)]
    if stratify is None:
        shuffled = rng.permutation(n_rows)
        for bucket, values in zip(buckets, np.array_split(shuffled, k)):
            bucket.extend(int(value) for value in values)
    else:
        labels = np.asarray(stratify)
        if labels.ndim != 1 or len(labels) != n_rows:
            raise ValueError(f"stratify must have shape ({n_rows},), got {labels.shape}")
        for label in np.unique(labels):
            rows = np.flatnonzero(labels == label)
            rng.shuffle(rows)
            for offset, row in enumerate(rows):
                buckets[offset % k].append(int(row))

    all_rows = np.arange(n_rows, dtype=np.int64)
    valid_count = max(1, int(round(n_rows * valid_fraction)))
    folds = []
    for fold, test_values in enumerate(buckets):
        test = np.asarray(sorted(test_values), dtype=np.int64)
        remaining = np.setdiff1d(all_rows, test, assume_unique=True)
        if valid_count >= len(remaining):
            raise ValueError(
                f"valid_fraction selects {valid_count} of {len(remaining)} non-test rows"
            )
        fold_rng = np.random.default_rng(seed + 10_000 + fold)
        if stratify is None:
            valid = np.sort(
                fold_rng.choice(remaining, size=valid_count, replace=False)
            ).astype(np.int64)
        else:
            labels = np.asarray(stratify)
            selected: list[int] = []
            for label in np.unique(labels[remaining]):
                candidates = remaining[labels[remaining] == label].copy()
                fold_rng.shuffle(candidates)
                count = int(round(valid_count * len(candidates) / len(remaining)))
                selected.extend(int(value) for value in candidates[:count])
            selected_set = set(selected)
            if len(selected) < valid_count:
                candidates = np.asarray(
                    [value for value in remaining if int(value) not in selected_set],
                    dtype=np.int64,
                )
                fold_rng.shuffle(candidates)
                selected.extend(int(value) for value in candidates[: valid_count - len(selected)])
            elif len(selected) > valid_count:
                selected = fold_rng.choice(
                    np.asarray(selected, dtype=np.int64),
                    size=valid_count,
                    replace=False,
                ).tolist()
            valid = np.asarray(sorted(selected), dtype=np.int64)
        train = np.setdiff1d(remaining, valid, assume_unique=True)
        combined = np.concatenate([train, valid, test])
        if len(combined) != n_rows or len(np.unique(combined)) != n_rows:
            raise RuntimeError(f"fold {fold + 1} is not a disjoint complete partition")
        folds.append((train, valid, test))
    return folds
