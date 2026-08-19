"""Dependency-light metrics used by the paper task implementations."""

from __future__ import annotations

import numpy as np


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def binary_statistics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    y_true, y_pred = np.asarray(y_true).astype(bool), np.asarray(y_pred).astype(bool)
    tp = int(np.sum(y_true & y_pred))
    tn = int(np.sum(~y_true & ~y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    precision, recall = safe_div(tp, tp + fp), safe_div(tp, tp + fn)
    return {
        "accuracy": safe_div(tp + tn, len(y_true)),
        "precision": precision,
        "recall": recall,
        "f1": safe_div(2 * precision * recall, precision + recall),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def multiclass_statistics(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int
) -> tuple[dict[str, float], dict[int, dict[str, float | int]], list[list[int]]]:
    y_true, y_pred = np.asarray(y_true, dtype=np.int64), np.asarray(y_pred, dtype=np.int64)
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(matrix, (y_true, y_pred), 1)
    per_class: dict[int, dict[str, float | int]] = {}
    f1_values = []
    for index in range(num_classes):
        stats = binary_statistics(y_true == index, y_pred == index)
        per_class[index] = {
            "precision": float(stats["precision"]),
            "recall": float(stats["recall"]),
            "f1": float(stats["f1"]),
            "support": int(np.sum(y_true == index)),
        }
        f1_values.append(float(stats["f1"]))
    accuracy = float(np.mean(y_true == y_pred)) if len(y_true) else 0.0
    sample_count = int(matrix.sum())
    true_count = matrix.sum(axis=1, dtype=np.float64)
    pred_count = matrix.sum(axis=0, dtype=np.float64)
    correct = float(np.trace(matrix))
    numerator = correct * sample_count - float(np.dot(true_count, pred_count))
    denominator = float(
        np.sqrt(
            (sample_count**2 - np.dot(pred_count, pred_count))
            * (sample_count**2 - np.dot(true_count, true_count))
        )
    )
    mcc = numerator / denominator if denominator > 0.0 else 0.0
    overall = {
        "accuracy": accuracy,
        "macro_f1": float(np.mean(f1_values)),
        "micro_f1": accuracy,
        "mcc": float(mcc),
    }
    return overall, per_class, matrix.tolist()


def regression_statistics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true, y_pred = np.asarray(y_true, dtype=np.float64), np.asarray(y_pred, dtype=np.float64)
    error = y_pred - y_true
    if y_true.ndim == 2:
        residual = np.sum(error ** 2, axis=0)
        centered = np.sum((y_true - np.mean(y_true, axis=0)) ** 2, axis=0)
        per_target_r2 = np.zeros_like(centered, dtype=np.float64)
        valid = centered > 0
        per_target_r2[valid] = 1.0 - residual[valid] / centered[valid]
        r2 = float(np.mean(per_target_r2))
    else:
        residual = float(np.sum(error ** 2))
        centered = float(np.sum((y_true - np.mean(y_true, axis=0)) ** 2))
        r2 = float(1.0 - residual / centered) if centered > 0 else 0.0
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "r2": r2,
    }


def normalized_regression_statistics(
    y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-10
) -> dict[str, float]:
    """Return range-normalized MAE/RMSE with equal weight per target."""
    truth = np.asarray(y_true, dtype=np.float64)
    prediction = np.asarray(y_pred, dtype=np.float64)
    if truth.ndim != 2 or prediction.shape != truth.shape:
        raise ValueError(
            "normalized regression statistics expect matching [N, K] arrays, "
            f"got {truth.shape} and {prediction.shape}"
        )
    ranges = np.maximum(truth.max(axis=0) - truth.min(axis=0), float(eps))
    error = prediction - truth
    normalized_mae = np.mean(np.abs(error), axis=0) / ranges
    normalized_rmse = np.sqrt(np.mean(error**2, axis=0)) / ranges
    return {
        "normalized_mae": float(np.mean(normalized_mae)),
        "normalized_rmse": float(np.mean(normalized_rmse)),
    }


def roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true, scores = np.asarray(y_true, dtype=np.int64), np.asarray(scores, dtype=np.float64)
    positive, negative = scores[y_true == 1], scores[y_true == 0]
    if not len(positive) or not len(negative):
        return 0.0
    comparisons = (positive[:, None] > negative[None]).sum()
    ties = (positive[:, None] == negative[None]).sum()
    return float((comparisons + 0.5 * ties) / (len(positive) * len(negative)))


def average_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(-np.asarray(scores))
    labels = np.asarray(y_true, dtype=np.int64)[order]
    positives = int(labels.sum())
    if positives == 0:
        return 0.0
    precision = np.cumsum(labels) / np.arange(1, len(labels) + 1)
    return float(np.sum(precision * labels) / positives)
