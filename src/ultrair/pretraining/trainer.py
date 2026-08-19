"""Single-process trainer for UltraIR multitask pretraining."""

from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from .augmentation import augment_signals
from .datasets import build_pretraining_loaders
from .losses import SoftTanimotoContrastiveLoss, WaveletReconstructionLoss
from .models import UltraIRPretrainingModel
from .utils.checkpoint import load_checkpoint, save_checkpoint, save_encoder
from .utils.metrics import embedding_tanimoto_spearman
from .utils.training import AverageMeter, build_warmup_cosine_lambda, set_seed


def _resolve_device(value: str) -> torch.device:
    if value.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(value)


def _build_model(config: dict[str, Any], signal_size: int, num_fgroups: int):
    model_cfg = config["model"]
    encoder_cfg = model_cfg["encoder"]
    wavelet_cfg = model_cfg["wavelet_head"]
    return UltraIRPretrainingModel(
        signal_size=signal_size,
        num_fgroups=num_fgroups,
        d_model=int(encoder_cfg.get("d_model", 1024)),
        patch_len=int(encoder_cfg.get("patch_len", 16)),
        n_heads=int(encoder_cfg.get("n_heads", 16)),
        num_global_layers=int(encoder_cfg.get("num_global_layers", 8)),
        dropout=float(encoder_cfg.get("dropout", 0.1)),
        head_dropout=float(encoder_cfg.get("head_dropout", 0.3)),
        fingerprint_proj_dim=int(model_cfg["fingerprint_head"].get("proj_dim", 256)),
        wavelet=str(wavelet_cfg.get("wavelet", "db4")),
        wavelet_level=int(wavelet_cfg.get("level", 4)),
        wavelet_hidden_dim=int(wavelet_cfg.get("hidden_dim", encoder_cfg.get("d_model", 1024))),
        wavelet_bottleneck_dim=(
            int(wavelet_cfg["bottleneck_dim"])
            if wavelet_cfg.get("bottleneck_dim") is not None
            else None
        ),
        wavelet_dropout=float(wavelet_cfg.get("dropout", 0.1)),
    )


def _autocast(device: torch.device, enabled: bool, dtype: torch.dtype):
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def _run_epoch(
    model: UltraIRPretrainingModel,
    loader,
    device: torch.device,
    config: dict[str, Any],
    reconstruction_loss: WaveletReconstructionLoss,
    fingerprint_loss: SoftTanimotoContrastiveLoss,
    functional_group_loss: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: LambdaLR | None = None,
    scaler: torch.amp.GradScaler | None = None,
    collect_embedding_metric: bool = False,
    max_batches: int | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    meters = {name: AverageMeter() for name in ("total", "reconstruction", "fingerprint", "functional_group")}
    metric_embeddings, metric_fingerprints = [], []
    weights = config["loss"]["weights"]
    use_amp = device.type == "cuda"
    use_bf16 = bool(config.get("use_bf16", True)) and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    grad_clip = float(config["train"].get("grad_clip", 1.0))

    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= int(max_batches):
            break
        signals = batch["signal"].to(device, non_blocking=True).float()
        fingerprints = batch["fingerprint"].to(device, non_blocking=True).float()
        groups = batch["functional_groups"].to(device, non_blocking=True).float()
        if training:
            inputs, reconstruction_mask = augment_signals(signals, config.get("augment", {}).get("shared", {}))
            optimizer.zero_grad(set_to_none=True)
        else:
            inputs, reconstruction_mask = signals, None

        with torch.set_grad_enabled(training):
            with _autocast(device, use_amp, amp_dtype):
                outputs = model(inputs)
                target_coefficients = model.wavelet_head.decompose(signals)
                loss_reconstruction, _ = reconstruction_loss(
                    outputs["reconstruction"], target_coefficients, signals, reconstruction_mask
                )
                loss_fingerprint = fingerprint_loss(
                    outputs["fingerprint_embedding"],
                    outputs["fingerprint_embedding"],
                    fingerprints,
                )
                loss_groups = functional_group_loss(outputs["fg_logits"], groups)
                total_loss = (
                    float(weights["reconstruction"]) * loss_reconstruction
                    + float(weights["fingerprint"]) * loss_fingerprint
                    + float(weights["functional_group"]) * loss_groups
                )

            if training:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(total_loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()
                if scheduler is not None:
                    scheduler.step()

        batch_size = signals.shape[0]
        meters["total"].update(float(total_loss.detach()), batch_size)
        meters["reconstruction"].update(float(loss_reconstruction.detach()), batch_size)
        meters["fingerprint"].update(float(loss_fingerprint.detach()), batch_size)
        meters["functional_group"].update(float(loss_groups.detach()), batch_size)
        if collect_embedding_metric:
            metric_embeddings.append(outputs["fingerprint_embedding"].detach().cpu())
            metric_fingerprints.append(fingerprints.detach().cpu())

    metrics = {name: meter.average for name, meter in meters.items()}
    if collect_embedding_metric and metric_embeddings:
        metric_cfg = config.get("metrics", {}).get("embedding_spearman", {})
        metrics["embedding_spearman"] = embedding_tanimoto_spearman(
            torch.cat(metric_embeddings),
            torch.cat(metric_fingerprints),
            max_pairs=int(metric_cfg.get("max_pairs", 500_000)),
            seed=int(metric_cfg.get("seed", config.get("seed", 42))),
        )
    return metrics


def run_pretraining(config: dict[str, Any]) -> dict[str, Any]:
    """Run the complete local pretraining, validation, and test workflow."""
    seed = int(config.get("seed", 42))
    set_seed(seed)
    device = _resolve_device(str(config.get("device", "cuda")))
    train_loader, valid_loader, test_loader, dataset_info, train_sampler = build_pretraining_loaders(config)
    if len(train_loader) == 0:
        raise ValueError("The training loader has no batches")

    model = _build_model(config, dataset_info.signal_size, dataset_info.num_fgroups).to(device)
    loss_cfg = config["loss"]
    reconstruction_criterion = WaveletReconstructionLoss(**loss_cfg["reconstruction"])
    fingerprint_criterion = SoftTanimotoContrastiveLoss(**loss_cfg["fingerprint"])
    functional_group_criterion = nn.BCEWithLogitsLoss()

    train_cfg = config["train"]
    epochs = int(train_cfg["epochs"])
    optimizer = AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 1e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
        betas=(0.9, 0.999),
    )
    total_steps = epochs * len(train_loader)
    scheduler = LambdaLR(
        optimizer,
        build_warmup_cosine_lambda(
            total_steps,
            int(total_steps * float(train_cfg.get("warmup_ratio", 0.05))),
            float(train_cfg.get("min_lr_ratio", 0.1)),
        ),
    )
    use_fp16_scaler = device.type == "cuda" and not (
        bool(config.get("use_bf16", True)) and torch.cuda.is_bf16_supported()
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16_scaler)

    save_dir = Path(train_cfg.get("save_dir", "checkpoints/pretraining")).expanduser()
    start_epoch, history, best_spearman = 1, [], float("-inf")
    resume_from = train_cfg.get("resume_from")
    if resume_from:
        payload = load_checkpoint(resume_from, model, optimizer, scheduler, device)
        start_epoch = int(payload.get("epoch", 0)) + 1
        history = list(payload.get("history", []))
        best_spearman = float(payload.get("best_embedding_spearman", float("-inf")))

    metric_cfg = config.get("metrics", {}).get("embedding_spearman", {})
    metric_enabled = bool(metric_cfg.get("enabled", True))
    max_eval_batches = metric_cfg.get("max_eval_batches")
    for epoch in range(start_epoch, epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch - 1)
        train_metrics = _run_epoch(
            model,
            train_loader,
            device,
            config,
            reconstruction_criterion,
            fingerprint_criterion,
            functional_group_criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
        )
        valid_metrics = _run_epoch(
            model,
            valid_loader,
            device,
            config,
            reconstruction_criterion,
            fingerprint_criterion,
            functional_group_criterion,
            collect_embedding_metric=metric_enabled,
            max_batches=max_eval_batches,
        )
        record = {"epoch": epoch, "train": train_metrics, "valid": valid_metrics}
        history.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        current_spearman = float(valid_metrics.get("embedding_spearman", -valid_metrics["total"]))
        is_best = current_spearman > best_spearman
        if is_best:
            best_spearman = current_spearman
        checkpoint = {
            "epoch": epoch,
            "config": config,
            "history": history,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_embedding_spearman": best_spearman,
        }
        save_checkpoint(save_dir, "last", checkpoint)
        if is_best:
            save_checkpoint(save_dir, "best", checkpoint)

    test_metrics = _run_epoch(
        model,
        test_loader,
        device,
        config,
        reconstruction_criterion,
        fingerprint_criterion,
        functional_group_criterion,
        collect_embedding_metric=metric_enabled,
        max_batches=max_eval_batches,
    )
    encoder_path = save_encoder(save_dir, epochs, model.encoder)
    summary = {
        "history": history,
        "test": test_metrics,
        "best_embedding_spearman": best_spearman,
        "encoder_checkpoint": str(encoder_path),
    }
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "history.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary
