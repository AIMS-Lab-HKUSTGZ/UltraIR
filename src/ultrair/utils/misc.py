"""Runtime, checkpoint, distributed, and metric-reporting helpers."""

from __future__ import annotations

import os
import random
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.distributed as dist


def ensure_dir(path: str | os.PathLike[str]) -> None:
    if path:
        Path(path).mkdir(parents=True, exist_ok=True)


_MEDICINAL_HERB_TASKS = {
    "medicinal_herb_constituent_quantification",
    "medicinal_herb_geographic_origin_traceability",
}
_MEDICINAL_HERB_DATASET_DIRS = {
    "jyh": "jyh",
    "jyh_lc": "jyh",
    "syh": "syh",
    "syh_lc": "syh",
}


def task_checkpoint_dir(config: dict[str, Any], task_name: str) -> Path:
    directory = Path(config.get("train", {}).get("save_root", "checkpoints")) / task_name
    if task_name not in _MEDICINAL_HERB_TASKS:
        return directory

    dataset = str(config.get("data", {}).get("dataset", "")).strip().lower()
    try:
        dataset_dir = _MEDICINAL_HERB_DATASET_DIRS[dataset]
    except KeyError as error:
        expected = ", ".join(sorted(_MEDICINAL_HERB_DATASET_DIRS))
        raise ValueError(
            f"Unsupported medicinal-herb dataset {dataset!r}; expected one of: {expected}"
        ) from error
    return directory / dataset_dir


def make_run_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S", time.localtime())


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def is_dist_available_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_dist_available_and_initialized() else 0


def is_main_process() -> bool:
    return get_rank() == 0


def barrier() -> None:
    if is_dist_available_and_initialized():
        dist.barrier()


def cleanup_distributed() -> None:
    if is_dist_available_and_initialized():
        dist.destroy_process_group()


def init_distributed_from_env(device: str, backend: Optional[str] = None) -> dict[str, Any]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    resolved_backend = backend
    if distributed and not is_dist_available_and_initialized():
        resolved_backend = backend or ("nccl" if device.startswith("cuda") else "gloo")
        if device.startswith("cuda"):
            torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=resolved_backend, init_method="env://")
    return {
        "distributed": distributed,
        "world_size": world_size,
        "rank": rank,
        "local_rank": local_rank,
        "backend": resolved_backend,
    }


def save_checkpoint(
    path: str | os.PathLike[str],
    model: torch.nn.Module,
    extra: Optional[dict[str, Any]] = None,
) -> None:
    ensure_dir(Path(path).parent)
    base_model = model.module if hasattr(model, "module") else model
    payload: dict[str, Any] = {"model": base_model.state_dict()}
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_model_state(
    model: torch.nn.Module,
    ckpt_path: str,
    map_location: str,
    strict: bool = True,
) -> None:
    checkpoint = torch.load(ckpt_path, map_location=map_location)
    state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state, dict):
        raise TypeError(f"Checkpoint state must be a mapping, got {type(state)}")
    if state and all(str(key).startswith("module.") for key in state):
        state = {str(key)[7:]: value for key, value in state.items()}
    base_model = model.module if hasattr(model, "module") else model
    model_keys = set(base_model.state_dict())
    if state and not model_keys.intersection(state) and hasattr(base_model, "encoder"):
        prefixed = {f"encoder.{key}": value for key, value in state.items()}
        if model_keys.intersection(prefixed):
            state = prefixed
    incompatible = base_model.load_state_dict(state, strict=strict)
    if not strict:
        if incompatible.missing_keys:
            print(f"[checkpoint] missing keys: {len(incompatible.missing_keys)}")
        if incompatible.unexpected_keys:
            print(f"[checkpoint] unexpected keys: {len(incompatible.unexpected_keys)}")


def _format_value(value: Any) -> str:
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6f}"
    return str(value)


def format_eval_txt(metrics: dict[str, Any], class_names: Optional[list[str]] = None) -> str:
    lines = ["===== Evaluation Metrics ====="]

    def visit(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                visit(f"{prefix}.{key}" if prefix else str(key), child)
        elif isinstance(value, (str, int, float, bool, np.generic)) or value is None:
            lines.append(f"{prefix}: {_format_value(value)}")

    visit("", metrics)
    if class_names:
        lines.append(f"class_names: {', '.join(map(str, class_names))}")
    return "\n".join(lines) + "\n"


def save_eval_txt(
    path: str | os.PathLike[str],
    metrics: dict[str, Any],
    class_names: Optional[list[str]] = None,
) -> str:
    text = format_eval_txt(metrics, class_names)
    ensure_dir(Path(path).parent)
    Path(path).write_text(text, encoding="utf-8")
    return text
