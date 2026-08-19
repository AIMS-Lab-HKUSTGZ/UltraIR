"""Small training-loop utilities."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class AverageMeter:
    total: float = 0.0
    count: int = 0

    def update(self, value: float, count: int = 1) -> None:
        self.total += float(value) * int(count)
        self.count += int(count)

    @property
    def average(self) -> float:
        return self.total / self.count if self.count else 0.0


def build_warmup_cosine_lambda(total_steps: int, warmup_steps: int, min_ratio: float = 0.1):
    total_steps, warmup_steps = max(1, int(total_steps)), max(0, int(warmup_steps))

    def schedule(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return float(step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))
        return float(min_ratio) + (1.0 - float(min_ratio)) * cosine

    return schedule
