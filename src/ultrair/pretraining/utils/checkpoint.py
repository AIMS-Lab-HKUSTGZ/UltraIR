"""Local checkpoint save and resume helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(directory: str | Path, name: str, payload: dict[str, Any]) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{name}.pt"
    temporary = directory / f".{name}.pt.tmp"
    torch.save(payload, temporary)
    temporary.replace(target)
    return target


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location=device)
    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError(f"Unsupported pretraining checkpoint: {path}")
    model.load_state_dict(payload["model"])
    if "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    if "scheduler" in payload:
        scheduler.load_state_dict(payload["scheduler"])
    return payload


def save_encoder(directory: str | Path, epoch: int, encoder: torch.nn.Module) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"epoch_{epoch}_encoder_no_task_head.pt"
    temporary = directory / f".{target.name}.tmp"
    torch.save(encoder.state_dict(), temporary)
    temporary.replace(target)
    return target
