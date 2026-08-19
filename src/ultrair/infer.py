"""Inference, evaluation, and optional prediction export."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from tqdm import tqdm

from ultrair.utils.misc import ensure_dir, format_eval_txt, is_main_process, save_eval_txt


def _move(value: Any, device: str) -> Any:
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move(child, device) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_move(child, device) for child in value)
    return value


def _merge(values: list[Any]) -> Any:
    if not values:
        return []
    first = values[0]
    if torch.is_tensor(first):
        return torch.cat([value.detach().cpu() for value in values])
    if isinstance(first, dict):
        return {key: _merge([value[key] for value in values]) for key in first}
    if isinstance(first, (list, tuple)):
        return [item for value in values for item in value]
    return values


@torch.no_grad()
def run_inference_collect(model, task, dataloader, device: str, use_bf16: bool):
    model.eval()
    predictions, targets, indices = [], [], []
    for batch in tqdm(dataloader, desc="Inference", leave=False, disable=not is_main_process()):
        if len(batch) == 2:
            signal, target = batch
            index = extras = None
        elif len(batch) == 3:
            signal, target, third = batch
            index, extras = (None, third) if isinstance(third, dict) else (third, None)
        elif len(batch) == 4:
            signal, target, index, extras = batch
        else:
            raise ValueError(f"Unsupported batch length: {len(batch)}")
        signal = signal.to(device).float()
        model_input: Any = signal
        if isinstance(extras, dict):
            model_input = {getattr(task, "ir_key", "ir"): signal, **_move(extras, device)}
        with torch.autocast(
            "cuda", dtype=torch.bfloat16 if use_bf16 else torch.float16,
            enabled=device.startswith("cuda"),
        ):
            prediction = task.forward_model(model, model_input)
        predictions.append(prediction.detach().float().cpu() if torch.is_tensor(prediction) else prediction)
        targets.append(target.detach().cpu() if torch.is_tensor(target) else target)
        if index is not None:
            indices.append(torch.as_tensor(index).cpu().long())
    return _merge(predictions), _merge(targets), _merge(indices) if indices else None


@torch.no_grad()
def run_prediction_collect(model, task, dataloader, device: str, use_bf16: bool):
    """Run inference for an unlabeled prediction loader."""

    model.eval()
    predictions, indices = [], []
    for batch in tqdm(dataloader, desc="Prediction", leave=False):
        if len(batch) == 2:
            signal, index = batch
            extras = None
        elif len(batch) == 3:
            signal, index, extras = batch
        else:
            raise ValueError(f"Unsupported prediction batch length: {len(batch)}")
        signal = signal.to(device).float()
        model_input: Any = signal
        if isinstance(extras, dict):
            model_input = {getattr(task, "ir_key", "ir"): signal, **_move(extras, device)}
        with torch.autocast(
            "cuda",
            dtype=torch.bfloat16 if use_bf16 else torch.float16,
            enabled=device.startswith("cuda"),
        ):
            prediction = task.forward_model(model, model_input)
        predictions.append(
            prediction.detach().float().cpu()
            if torch.is_tensor(prediction)
            else prediction
        )
        indices.append(torch.as_tensor(index).cpu().long())
    return _merge(predictions), _merge(indices)


def _jsonable(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(child) for child in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _prediction_arrays(task, predictions: torch.Tensor, targets: torch.Tensor, metrics):
    logits = predictions.detach().cpu().numpy()
    truth = targets.detach().cpu().numpy()
    overall = metrics.get("overall", {}) or {}
    payload: dict[str, Any] = {"logits": logits, "targets": truth}
    if overall.get("is_regression"):
        payload.update(preds=logits, prediction_type=np.array("regression"))
    elif overall.get("is_multiclass"):
        shifted = logits - logits.max(axis=1, keepdims=True)
        probabilities = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
        payload.update(probs=probabilities, preds=probabilities.argmax(1), prediction_type=np.array("multiclass"))
    else:
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        threshold = float(metrics.get("threshold", overall.get("threshold", 0.5)))
        payload.update(
            probs=probabilities,
            preds=(probabilities >= threshold).astype(np.int8),
            threshold=np.array(threshold),
            prediction_type=np.array("binary" if overall.get("is_binary") else "multilabel"),
        )
    return payload


def _save_prediction_csv(path: Path, payload: dict[str, Any], sample_indices: Optional[torch.Tensor]) -> None:
    predictions, targets = np.asarray(payload["preds"]), np.asarray(payload["targets"])
    indices = sample_indices.numpy() if sample_indices is not None else np.arange(len(predictions))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row_id", "sample_index", "target", "prediction"])
        for row, (target, prediction) in enumerate(zip(targets, predictions)):
            writer.writerow([
                row, int(indices[row]),
                json.dumps(np.asarray(target).tolist()),
                json.dumps(np.asarray(prediction).tolist()),
            ])


def infer_eval_and_save_txt(
    cfg: dict[str, Any], model, task, dataloader, method_name: str, fold: int,
    split: str, device: str, use_bf16: bool,
    class_names: Optional[list[str]] = None, ckpt_path: Optional[str] = None,
    tag: Optional[str] = None, metrics_sink: Optional[list[dict[str, Any]]] = None,
) -> Optional[str]:
    predictions, targets, indices = run_inference_collect(model, task, dataloader, device, use_bf16)
    needs_indices = bool((cfg.get("task", {}).get("eval", {}) or {}).get("needs_sample_indices", False))
    metrics = (
        task.eval_from_logits_and_targets(predictions, targets, sample_indices=indices)
        if needs_indices else task.eval_from_logits_and_targets(predictions, targets)
    )
    tag = tag or ("no_ckpt" if ckpt_path is None else Path(ckpt_path).stem)
    if metrics_sink is not None:
        metrics_sink.append({
            "fold": fold,
            "checkpoint": ckpt_path,
            "checkpoint_tag": tag,
            "metrics": metrics,
        })
    settings = cfg.get("results", {}) or {}
    root = Path(settings.get("root", "results")) / task.name / method_name / f"fold-{fold}"
    save_txt = bool(settings.get("save_txt", True))
    save_json = bool(settings.get("save_json", True))
    save_predictions = bool(settings.get("save_predictions", True))
    if save_txt or save_json or save_predictions:
        ensure_dir(root)
    text_path = root / f"{split}_{tag}.txt"
    text = save_eval_txt(text_path, metrics, class_names) if save_txt else format_eval_txt(metrics, class_names)
    if save_json:
        (root / f"{split}_{tag}.json").write_text(
            json.dumps({
                "metrics": _jsonable(metrics), "checkpoint": ckpt_path,
                "method": method_name, "task": task.name, "fold": fold, "split": split,
            }, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    if save_predictions and torch.is_tensor(predictions) and torch.is_tensor(targets):
        payload = _prediction_arrays(task, predictions, targets, metrics)
        if indices is not None:
            payload["sample_indices"] = indices.numpy()
        if class_names:
            payload["class_names"] = np.asarray(class_names, dtype=object)
        np.savez_compressed(root / f"{split}_{tag}_predictions.npz", **payload)
        _save_prediction_csv(root / f"{split}_{tag}_sample_predictions.csv", payload, indices)
    if is_main_process():
        print(text, end="")
    return str(text_path) if save_txt else None
