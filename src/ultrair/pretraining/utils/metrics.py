"""Embedding metrics used for validation and testing."""

from __future__ import annotations

import math

import numpy as np
import torch


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2:
        return 0.0
    left_rank, right_rank = _average_ranks(left), _average_ranks(right)
    left_rank, right_rank = left_rank - left_rank.mean(), right_rank - right_rank.mean()
    denominator = math.sqrt(float(left_rank @ left_rank) * float(right_rank @ right_rank))
    return float(left_rank @ right_rank / denominator) if denominator > 0 else 0.0


def embedding_tanimoto_spearman(
    embeddings: torch.Tensor,
    fingerprints: torch.Tensor,
    max_pairs: int = 500_000,
    seed: int = 42,
) -> float:
    if embeddings.ndim != 2 or fingerprints.ndim != 2:
        raise ValueError("Embeddings and fingerprints must both be two-dimensional")
    if embeddings.shape[0] != fingerprints.shape[0]:
        raise ValueError("Embeddings and fingerprints must have equal row counts")
    count = embeddings.shape[0]
    if count < 2:
        return 0.0
    total_pairs = count * (count - 1) // 2
    if max_pairs <= 0 or total_pairs <= max_pairs:
        rows, columns = np.triu_indices(count, k=1)
    else:
        flat = np.sort(np.random.default_rng(seed).choice(total_pairs, max_pairs, replace=False))
        offsets = np.concatenate(([0], np.cumsum(np.arange(count - 1, 0, -1, dtype=np.int64))))
        rows = np.searchsorted(offsets[1:], flat, side="right")
        columns = flat - offsets[rows] + rows + 1
    normalized = torch.nn.functional.normalize(embeddings.detach().float(), dim=1).cpu().numpy()
    bits = (fingerprints.detach().float().cpu().numpy() > 0.5).astype(np.float32)
    cosine = (normalized[rows] * normalized[columns]).sum(axis=1)
    intersection = (bits[rows] * bits[columns]).sum(axis=1)
    union = bits[rows].sum(axis=1) + bits[columns].sum(axis=1) - intersection
    return _spearman(cosine, intersection / np.maximum(union, 1e-8))
