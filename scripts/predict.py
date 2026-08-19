"""Predict from unlabeled spectra without a test split or fold-layout input."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from scripts.run import (
    build_eval_transform,
    load_yaml,
    parse_fold,
    resolve_config_path,
    resolve_initial_checkpoint,
    resolve_model_config,
)
from ultrair.datasets.npy_dataset import build_prediction_loader
from ultrair.infer import run_prediction_collect
from ultrair.models.registry import build_model
from ultrair.tasks.registry import build_task
from ultrair.utils.misc import load_model_state, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        help="Task YAML config; takes precedence when --task is also provided",
    )
    parser.add_argument(
        "--task",
        help="Task name or unambiguous initialism for a task with one config",
    )
    parser.add_argument("--ckpt", type=Path, required=True, help="Task checkpoint")
    parser.add_argument(
        "--ckpt-tag",
        choices=["best", "last"],
        help="Checkpoint name to select when --ckpt is a directory",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Unlabeled spectrum NPY file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Destination JSON file; omit to print JSON to standard output",
    )
    formula = parser.add_mutually_exclusive_group()
    formula.add_argument(
        "--formula",
        type=Path,
        help="Formula NPY aligned with --input (structure elucidation only)",
    )
    formula.add_argument(
        "--formula-text",
        help="One molecular formula to apply to every input spectrum",
    )
    parser.add_argument(
        "--stats-fold",
        type=parse_fold,
        help="Reference fold for configured training statistics (default: YAML default_fold)",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        help="Override config.data.root for reference statistics",
    )
    parser.add_argument("--device", help="Device override, for example cuda:0 or cpu")
    parser.add_argument("--batch-size", type=int, help="Prediction batch size")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--threshold",
        type=float,
        help="Probability threshold for functional-group or detection output",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        help="Beam size override for molecular structure generation",
    )
    parser.add_argument(
        "--num-candidates",
        type=int,
        help="Number of ranked SMILES to return per spectrum",
    )
    parser.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require an exact checkpoint/model match (default: true)",
    )
    args = parser.parse_args()
    if args.config is None and args.task is None:
        parser.error("one of --config or --task is required")
    return args


def _resolve_device(cfg: dict[str, Any], override: str | None) -> str:
    device = str(
        override
        or cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    )
    if device.startswith("cuda") and not torch.cuda.is_available():
        print(
            f"[predict] CUDA is unavailable; using CPU instead of {device!r}",
            file=sys.stderr,
        )
        return "cpu"
    return device


def _names_or_indices(task: Any, width: int) -> list[str]:
    names = task.class_names()
    if names is None:
        return [f"class_{index}" for index in range(width)]
    if len(names) != width:
        raise ValueError(
            f"Task defines {len(names)} output names, but the model returned {width} values"
        )
    return [str(name) for name in names]


def _rows_from_predictions(
    task: Any,
    predictions: Any,
    indices: torch.Tensor,
    threshold_override: float | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    row_indices = [int(value) for value in indices.tolist()]
    task_name = task.name

    if task_name == "molecular_structure_elucidation":
        if not isinstance(predictions, dict) or "topk_smiles" not in predictions:
            raise ValueError("Structural prediction did not return ranked SMILES")
        candidates = predictions["topk_smiles"]
        rows = [
            {
                "row_index": row_index,
                "smiles_candidates": [str(value) for value in values],
            }
            for row_index, values in zip(row_indices, candidates)
        ]
        return rows, {"prediction_type": "ranked_smiles"}

    if not torch.is_tensor(predictions):
        raise TypeError(
            f"Task {task_name!r} returned unsupported predictions: {type(predictions)}"
        )
    values = predictions.detach().cpu().float()

    if task_name == "targeted_component_detection":
        threshold = float(
            task.threshold if threshold_override is None else threshold_override
        )
        probabilities = torch.sigmoid(values).view(-1).numpy()
        rows = []
        for row_index, probability in zip(row_indices, probabilities):
            present = bool(float(probability) >= threshold)
            rows.append(
                {
                    "row_index": row_index,
                    "present_probability": float(probability),
                    "class_index": int(present),
                    "class_name": "present" if present else "absent",
                }
            )
        return rows, {"prediction_type": "binary", "threshold": threshold}

    if task_name == "targeted_fractional_contribution_estimation":
        estimates = values.view(-1).numpy()
        rows = [
            {
                "row_index": row_index,
                str(task.target_name): float(estimate),
            }
            for row_index, estimate in zip(row_indices, estimates)
        ]
        return rows, {"prediction_type": "regression"}

    if task_name == "functional_group_prediction":
        settings = task.threshold_search or {}
        threshold = float(
            settings.get("fixed_threshold", 0.5)
            if threshold_override is None
            else threshold_override
        )
        probabilities = torch.sigmoid(values).numpy()
        names = _names_or_indices(task, probabilities.shape[1])
        rows = []
        for row_index, probability in zip(row_indices, probabilities):
            selected = [
                name
                for name, score in zip(names, probability)
                if float(score) >= threshold
            ]
            rows.append(
                {
                    "row_index": row_index,
                    "selected_labels": selected,
                    "probabilities": {
                        name: float(score) for name, score in zip(names, probability)
                    },
                }
            )
        return rows, {
            "prediction_type": "multilabel",
            "threshold": threshold,
            "class_names": names,
        }

    if hasattr(task, "denormalize"):
        denormalized = task.denormalize(values.numpy())
        names = _names_or_indices(task, denormalized.shape[1])
        rows = [
            {
                "row_index": row_index,
                "values": {
                    name: float(value) for name, value in zip(names, prediction)
                },
            }
            for row_index, prediction in zip(row_indices, denormalized)
        ]
        return rows, {
            "prediction_type": "regression",
            "target_names": names,
            "scale": "original",
        }

    probabilities = torch.softmax(values, dim=1).numpy()
    names = _names_or_indices(task, probabilities.shape[1])
    rows = []
    for row_index, probability in zip(row_indices, probabilities):
        class_index = int(np.argmax(probability))
        rows.append(
            {
                "row_index": row_index,
                "class_index": class_index,
                "class_name": names[class_index],
                "probabilities": {
                    name: float(score) for name, score in zip(names, probability)
                },
            }
        )
    return rows, {"prediction_type": "multiclass", "class_names": names}


def main() -> None:
    args = parse_args()
    config_path = resolve_config_path(args.config, args.task)
    cfg = load_yaml(config_path)
    if args.data_root is not None:
        cfg.setdefault("data", {})["root"] = str(args.data_root.expanduser())
    device = _resolve_device(cfg, args.device)
    set_seed(int(cfg.get("seed", 42)))

    task_section = cfg.get("task", {}) or {}
    task_name = task_section.get("name")
    if not task_name:
        raise ValueError("config.task.name is required")
    task = build_task(task_name, task_section.get("args", {}) or {})

    is_structural = task.name == "molecular_structure_elucidation"
    if is_structural and args.formula is None and args.formula_text is None:
        raise ValueError(
            "Molecular structure elucidation requires --formula or --formula-text"
        )
    if not is_structural and (args.formula is not None or args.formula_text is not None):
        raise ValueError("--formula and --formula-text apply only to structure elucidation")
    if args.threshold is not None and not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be between 0 and 1")
    if args.threshold is not None and task.name not in {
        "functional_group_prediction",
        "targeted_component_detection",
    }:
        raise ValueError(
            "--threshold applies only to functional-group prediction and "
            "targeted component detection"
        )
    if args.beam_size is not None:
        if not is_structural or args.beam_size < 1:
            raise ValueError("--beam-size must be positive and requires structure elucidation")
        task.beam_size = int(args.beam_size)
    if args.num_candidates is not None:
        if not is_structural or args.num_candidates < 1:
            raise ValueError(
                "--num-candidates must be positive and requires structure elucidation"
            )
        task.num_return_sequences = int(args.num_candidates)
        task.beam_size = max(int(task.beam_size), int(task.num_return_sequences))

    model_name, model_cfg = resolve_model_config(cfg)
    if "signal_size" not in model_cfg:
        raise ValueError(f"config.model configuration for {model_name!r} needs signal_size")
    augment_cfg = cfg.setdefault("augment", {})
    augment_cfg["eval_transform"] = build_eval_transform(
        int(model_cfg["signal_size"]),
        legacy_semantics=bool(augment_cfg.get("legacy_semantics", False)),
    )

    stats_fold = args.stats_fold
    if stats_fold is None:
        stats_fold = parse_fold(str(cfg["data"].get("default_fold", "demo")))
    extra_inputs = None
    if is_structural:
        formulas = (
            np.load(args.formula.expanduser(), allow_pickle=True)
            if args.formula is not None
            else np.asarray(args.formula_text)
        )
        extra_inputs = {task.formula_key: formulas}

    loader, dataset_info = build_prediction_loader(
        cfg,
        task,
        args.input.expanduser(),
        stats_fold=stats_fold,
        extra_inputs=extra_inputs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    model = build_model(model_name, model_cfg, dataset_info).to(device)
    checkpoint_tag = str(
        args.ckpt_tag or cfg.get("run", {}).get("ckpt_tag", "last")
    )
    checkpoint_path = resolve_initial_checkpoint(args.ckpt, tag=checkpoint_tag)
    load_model_state(
        model,
        str(checkpoint_path),
        map_location=device,
        strict=bool(args.strict),
    )
    predictions, indices = run_prediction_collect(
        model,
        task,
        loader,
        device,
        use_bf16=bool(cfg.get("use_bf16", True)),
    )
    rows, prediction_meta = _rows_from_predictions(
        task, predictions, indices, args.threshold
    )
    payload = {
        "task": task.name,
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "input": str(args.input.expanduser()),
        "stats_fold": stats_fold,
        "num_samples": len(rows),
        **prediction_meta,
        "predictions": rows,
    }
    rendered = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.output is None:
        print(rendered)
    else:
        output_path = args.output.expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
        print(f"saved: {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
