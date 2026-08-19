"""Fingerprint-supervised embedding loss."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def tanimoto_similarity_matrix(fingerprints: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    fingerprints = (fingerprints > 0.5).float()
    intersection = fingerprints @ fingerprints.t()
    cardinality = fingerprints.sum(dim=1, keepdim=True)
    union = cardinality + cardinality.t() - intersection
    return intersection / (union + eps)


def _mask_diagonal(matrix: torch.Tensor, value: float) -> torch.Tensor:
    output = matrix.clone()
    diagonal = torch.arange(output.shape[0], device=output.device)
    output[diagonal, diagonal] = value
    return output


def _soft_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    log_probabilities = F.log_softmax(logits, dim=1)
    log_probabilities = torch.where(targets > 0, log_probabilities, torch.zeros_like(log_probabilities))
    return -(targets * log_probabilities).sum(dim=1).mean()


class SoftTanimotoContrastiveLoss(nn.Module):
    def __init__(
        self,
        student_temperature: float = 0.1,
        target_temperature: float = 0.05,
        symmetric: bool = True,
    ) -> None:
        super().__init__()
        self.student_temperature = float(student_temperature)
        self.target_temperature = float(target_temperature)
        self.symmetric = bool(symmetric)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        fingerprints: torch.Tensor,
    ) -> torch.Tensor:
        if query.shape[0] <= 1:
            return query.new_zeros(())
        same_inputs = query.data_ptr() == key.data_ptr()
        query, key = F.normalize(query, dim=1), F.normalize(key, dim=1)
        targets = tanimoto_similarity_matrix(fingerprints)
        query_logits = query @ key.t()
        if same_inputs:
            targets = _mask_diagonal(targets, float("-inf"))
            query_logits = _mask_diagonal(query_logits, float("-inf"))
        query_targets = F.softmax(targets / self.target_temperature, dim=1)
        query_loss = _soft_cross_entropy(query_logits / self.student_temperature, query_targets)
        if not self.symmetric or same_inputs:
            return query_loss
        key_targets = F.softmax(targets.t() / self.target_temperature, dim=1)
        key_logits = key @ query.t()
        key_loss = _soft_cross_entropy(key_logits / self.student_temperature, key_targets)
        return 0.5 * (query_loss + key_loss)
