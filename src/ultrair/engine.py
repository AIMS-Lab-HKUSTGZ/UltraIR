"""Standard PyTorch training and validation loop for UltraIR tasks."""

from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from ultrair.infer import run_inference_collect
from ultrair.utils.misc import (
    barrier,
    ensure_dir,
    is_dist_available_and_initialized,
    is_main_process,
    save_checkpoint,
    task_checkpoint_dir,
)


def _move(value: Any, device: str) -> Any:
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move(child, device) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_move(child, device) for child in value)
    return value


def _unpack_batch(batch: Any):
    if not isinstance(batch, (tuple, list)):
        raise TypeError(f"Expected tuple/list batch, got {type(batch)}")
    if len(batch) == 2:
        return batch[0], batch[1], None
    if len(batch) == 3:
        return batch[0], batch[1], batch[2] if isinstance(batch[2], dict) else None
    if len(batch) == 4:
        return batch[0], batch[1], batch[3]
    raise ValueError(f"Unsupported batch length: {len(batch)}")


def _prepare_targets(task: Any, targets: Any, device: str) -> Any:
    targets = _move(targets, device)
    prepare = getattr(task, "prepare_targets", None)
    if callable(prepare):
        targets = prepare(targets)
    elif torch.is_tensor(targets):
        targets = targets.float()
    return _move(targets, device)


def _model_input(task: Any, signal: torch.Tensor, extras: Any, targets: Any = None) -> Any:
    ir_key = getattr(task, "ir_key", "ir")
    values: Any = signal
    if isinstance(extras, dict):
        values = {ir_key: signal, **extras}
    if getattr(task, "forward_needs_targets", False):
        if not isinstance(values, dict):
            values = {ir_key: signal}
        values[getattr(task, "target_key", "targets")] = targets
    return values


def _task_loss(task: Any, criterion, output: Any, targets: Any, model_input: Any, model):
    compute = getattr(task, "compute_loss", None)
    if callable(compute):
        return compute(
            output,
            targets,
            criterion=criterion,
            batch_input=model_input,
            model=model,
        )
    return criterion(output, targets)


def _reduce_average(total: float, count: int, device: str) -> float:
    if is_dist_available_and_initialized():
        values = torch.tensor([total, count], dtype=torch.float64, device=device)
        dist.all_reduce(values)
        total, count = float(values[0]), int(values[1])
    return total / max(count, 1)


@torch.no_grad()
def evaluate_loss(model, task, dataloader, criterion, device: str, use_bf16: bool) -> float:
    model.eval()
    total, count = 0.0, 0
    for batch in dataloader:
        signal, targets, extras = _unpack_batch(batch)
        signal = signal.to(device).float()
        extras = _move(extras, device)
        targets = _prepare_targets(task, targets, device)
        values = _model_input(task, signal, extras, targets)
        with torch.autocast(
            "cuda", dtype=torch.bfloat16 if use_bf16 else torch.float16,
            enabled=device.startswith("cuda"),
        ):
            output = task.forward_model(model, values)
            loss = _task_loss(task, criterion, output, targets, values, model)
        total, count = total + float(loss.detach()), count + 1
    return _reduce_average(total, count, device)


def _metric_at_path(metrics: dict[str, Any], selector: str) -> float:
    value: Any = metrics
    for key in selector.split("."):
        value = value[key]
    return float(value)


def evaluate_metric(model, task, dataloader, selector: str, device: str, use_bf16: bool) -> float:
    predictions, targets, indices = run_inference_collect(model, task, dataloader, device, use_bf16)
    select_threshold = getattr(task, "select_eval_threshold", None)
    if callable(select_threshold):
        select_threshold(predictions, targets)
    metrics = (
        task.eval_from_logits_and_targets(predictions, targets, sample_indices=indices)
        if indices is not None else task.eval_from_logits_and_targets(predictions, targets)
    )
    return _metric_at_path(metrics, selector)


def _scheduler(optimizer, total_steps: int, warmup_steps: int, min_lr: float, base_lr: float):
    min_ratio = min(float(min_lr) / base_lr, 1.0) if base_lr > 0 else 0.0

    def factor(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(max(progress, 0), 1)))

    return LambdaLR(optimizer, factor)


def _freeze(model: torch.nn.Module, policy: str) -> None:
    policy = policy.lower()
    if policy in {"", "none", "all"}:
        return
    base = model.module if hasattr(model, "module") else model
    for parameter in base.parameters():
        parameter.requires_grad = False
    prefixes = {
        "head": ("backbone.classifier", "classifier"),
        "head_only": ("backbone.classifier", "classifier"),
        "head_fusion": ("backbone.classifier", "classifier", "input_fusion"),
        "encoder": ("derivative_module", "input_fusion", "backbone"),
    }.get(policy)
    if prefixes is None:
        raise ValueError(f"Unknown freeze policy: {policy}")
    for name, module in base.named_modules():
        if any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes):
            for parameter in module.parameters():
                parameter.requires_grad = True


def train_and_validate(
    cfg: dict[str, Any],
    model,
    task,
    train_dl,
    val_dl,
    device: str,
    use_bf16: bool,
    method_name: str,
    fold: int,
    run_id: str,
) -> dict[str, Optional[str]]:
    train_cfg = cfg.get("train", {}) or {}
    epochs = int(train_cfg.get("epochs", 50))
    lr = float((train_cfg.get("lr_by_model", {}) or {}).get(method_name, train_cfg.get("lr", 3e-4)))
    weight_decay = float(train_cfg.get("weight_decay", 1e-2))
    _freeze(model, str(train_cfg.get("freeze_policy", "none")))
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("No trainable model parameters")
    criterion = task.build_criterion().to(device)
    optimizer = AdamW(parameters, lr=lr, weight_decay=weight_decay)
    total_steps = max(1, epochs * len(train_dl))
    scheduler_cfg = train_cfg.get("scheduler", {}) or {}
    warmup_steps = int(scheduler_cfg.get("warmup_steps", 0))
    if not warmup_steps:
        warmup_steps = int(scheduler_cfg.get("warmup_epochs", 0)) * len(train_dl)
    scheduler = _scheduler(
        optimizer, total_steps, warmup_steps,
        float(scheduler_cfg.get("eta_min", 0.0)), lr,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.startswith("cuda") and not use_bf16)
    best_selector = str(train_cfg.get("best_selector", "val_loss"))
    best_mode = str(train_cfg.get("best_mode", "min" if best_selector == "val_loss" else "max"))
    best_score = float("inf") if best_mode == "min" else float("-inf")
    best_path: Optional[str] = None
    last_path: Optional[str] = None
    checkpoint_dir = (
        task_checkpoint_dir(cfg, task.name)
        / method_name
        / f"fold-{fold}"
    )
    should_save = bool(train_cfg.get("save_best", True) or train_cfg.get("save_last", True) or train_cfg.get("save_every", 0))
    if should_save:
        ensure_dir(checkpoint_dir)

    for epoch in range(1, epochs + 1):
        model.train()
        sampler = getattr(train_dl, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        total, count = 0.0, 0
        progress = tqdm(train_dl, desc=f"epoch {epoch}/{epochs}", leave=False, disable=not is_main_process())
        for batch in progress:
            signal, targets, extras = _unpack_batch(batch)
            signal, extras = signal.to(device).float(), _move(extras, device)
            targets = _prepare_targets(task, targets, device)
            values = _model_input(task, signal, extras, targets)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                "cuda", dtype=torch.bfloat16 if use_bf16 else torch.float16,
                enabled=device.startswith("cuda"),
            ):
                output = task.forward_model(model, values)
                loss = _task_loss(task, criterion, output, targets, values, model)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if float(train_cfg.get("grad_clip_norm", 0)) > 0:
                    torch.nn.utils.clip_grad_norm_(parameters, float(train_cfg["grad_clip_norm"]))
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if float(train_cfg.get("grad_clip_norm", 0)) > 0:
                    torch.nn.utils.clip_grad_norm_(parameters, float(train_cfg["grad_clip_norm"]))
                optimizer.step()
            scheduler.step()
            total, count = total + float(loss.detach()), count + 1
            progress.set_postfix(loss=float(loss.detach()), lr=optimizer.param_groups[0]["lr"])

        train_loss = _reduce_average(total, count, device)
        val_loss = evaluate_loss(model, task, val_dl, criterion, device, use_bf16)
        score = val_loss if best_selector == "val_loss" else evaluate_metric(
            model, task, val_dl, best_selector, device, use_bf16
        )
        if is_main_process():
            print(f"[fold-{fold}] epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f} {best_selector}={score:.6f}")
        metadata = {
            "task": task.name, "method": method_name, "fold": fold, "run_id": run_id,
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            "selected_score": score,
        }
        if train_cfg.get("save_last", True) and is_main_process():
            last_path = str(checkpoint_dir / f"last_{run_id}.pt")
            save_checkpoint(last_path, model, metadata)
        interval = int(train_cfg.get("save_every", 0))
        if interval > 0 and epoch % interval == 0 and is_main_process():
            save_checkpoint(checkpoint_dir / f"epoch_{epoch:03d}_{run_id}.pt", model, metadata)
        improved = score < best_score if best_mode == "min" else score > best_score
        if train_cfg.get("save_best", True) and improved and is_main_process():
            best_score = score
            best_path = str(checkpoint_dir / f"best_{run_id}.pt")
            save_checkpoint(best_path, model, metadata)
    barrier()
    return {"last": last_path, "best": best_path}
