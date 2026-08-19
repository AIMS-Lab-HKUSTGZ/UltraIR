"""Train and evaluate UltraIR on one fold or a complete cross-validation run."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from numbers import Real
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader

from ultrair.datasets.npy_dataset import build_loaders_for_task
from ultrair.engine import train_and_validate
from ultrair.infer import infer_eval_and_save_txt, run_inference_collect
from ultrair.models.registry import build_model
from ultrair.tasks.registry import TASK_NAMES, build_task
from ultrair.utils.misc import (
    cleanup_distributed,
    ensure_dir,
    is_main_process,
    load_model_state,
    make_run_id,
    set_seed,
    task_checkpoint_dir,
)
from ultrair.utils.transforms import AddNoise, Resizer, ShiftLR, ShiftUD


class Compose:
    def __init__(self, transforms: list[Any]) -> None:
        self.transforms = transforms

    def __call__(self, sample: Any) -> Any:
        for transform in self.transforms:
            sample = transform(sample)
        return sample


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return data


def build_train_transform(
    aug_cfg: dict[str, Any], signal_size: int, *, legacy_semantics: bool = False
) -> Compose:
    return Compose(
        [
            AddNoise(
                p=float(aug_cfg.get("p_noise", 0.0)),
                target_snr_db=aug_cfg.get("noise_db", [2, 20]),
                mean_noise=float(aug_cfg.get("mean_noise", 0.0)),
                legacy_semantics=legacy_semantics,
            ),
            ShiftLR(
                p=float(aug_cfg.get("p_shiftLR", 0.0)),
                shift_p=aug_cfg.get("shiftLR_size", [0.01, 0.1]),
            ),
            ShiftUD(
                p=float(aug_cfg.get("p_shiftUD", 0.0)),
                shift_p=aug_cfg.get("shiftUD_size", [0.01, 0.1]),
                legacy_semantics=legacy_semantics,
            ),
            Resizer(signal_size=signal_size, legacy_semantics=legacy_semantics),
        ]
    )


def build_eval_transform(signal_size: int, *, legacy_semantics: bool = False) -> Compose:
    return Compose(
        [Resizer(signal_size=signal_size, legacy_semantics=legacy_semantics)]
    )


def resolve_model_config(cfg: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    model_section = cfg.get("model", {}) or {}
    model_name = model_section.get("name")
    if not model_name:
        raise ValueError("config.model.name is required")

    configs = model_section.get("configs")
    if isinstance(configs, dict) and configs:
        if model_name not in configs:
            raise ValueError(
                f"config.model.name={model_name!r} is not present in "
                f"config.model.configs ({', '.join(configs)})"
            )
        model_cfg = configs[model_name] or {}
    else:
        model_cfg = model_section.get("args", {}) or {}

    if not isinstance(model_cfg, dict):
        raise ValueError(f"Configuration for model {model_name!r} must be a mapping")
    return str(model_name), dict(model_cfg)


Fold = int | str

COMPLETE_PACKAGED_DEMO_TASKS = {
    "medicinal_herb_constituent_quantification",
    "medicinal_herb_geographic_origin_traceability",
}


def parse_fold(value: str) -> Fold:
    normalized = value.strip().lower()
    if normalized in {"demo", "fold-demo"}:
        return "demo"
    try:
        return int(normalized)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("fold must be 1-5 or 'demo'") from exc


def parse_bool(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError("expected a boolean value")


def task_initialism(task_name: str) -> str:
    return "".join(part[0] for part in task_name.split("_") if part)


def normalize_task_identifier(value: str) -> str:
    return "_".join(
        part
        for part in value.strip().lower().replace("-", "_").replace(" ", "_").split("_")
        if part
    )


def resolve_task_name(identifier: str) -> str:
    normalized = normalize_task_identifier(identifier)
    matches = sorted(
        task_name
        for task_name in TASK_NAMES
        if normalized in {task_name, task_initialism(task_name)}
    )
    if not matches:
        choices = ", ".join(
            f"{task_name} ({task_initialism(task_name)})"
            for task_name in sorted(TASK_NAMES)
        )
        raise ValueError(f"Unknown task {identifier!r}. Available tasks: {choices}")
    if len(matches) > 1:
        raise ValueError(
            f"Ambiguous task abbreviation {identifier!r}; matches: {', '.join(matches)}"
        )
    return matches[0]


def resolve_task_config(identifier: str, config_root: Path | None = None) -> Path:
    task_name = resolve_task_name(identifier)
    root = config_root or Path(__file__).resolve().parents[1] / "configs"
    task_dir = root / task_name
    candidates = sorted(task_dir.glob("*.yaml"))
    if not candidates:
        raise FileNotFoundError(f"No YAML config found for task {task_name!r} in {task_dir}")
    if len(candidates) == 1:
        return candidates[0]
    names = ", ".join(path.name for path in candidates)
    raise ValueError(
        f"Task {task_name!r} has multiple configs: {names}. "
        "Select the dataset explicitly with --config."
    )


def resolve_config_path(config: Path | None, task: str | None) -> Path:
    if config is not None:
        return config.expanduser().resolve()
    if task is None:
        raise ValueError("Either --config or --task is required")
    return resolve_task_config(task).resolve()


def checkpoint_dir(cfg: dict[str, Any], task: Any, method_name: str, fold: Fold) -> Path:
    return task_checkpoint_dir(cfg, task.name) / method_name / f"fold-{fold}"


def latest_checkpoint(directory: Path, tag: str) -> Path | None:
    if not directory.is_dir():
        return None
    exact_name = f"{tag}.pt"
    candidates = sorted(
        [*directory.glob(f"{tag}_*.pt"), directory / exact_name],
        key=lambda path: (
            path.stat().st_mtime_ns if path.is_file() else -1,
            path.name == exact_name,
        ),
        reverse=True,
    )
    return next((path for path in candidates if path.is_file()), None)


def resolve_initial_checkpoint(path_value: str | Path, tag: str = "last") -> Path:
    path = Path(path_value).expanduser()
    if path.is_dir():
        resolved = latest_checkpoint(path, tag)
        if resolved is None:
            raise FileNotFoundError(f"No {tag!r} checkpoint found in {path}")
        return resolved
    if not path.is_file():
        raise FileNotFoundError(f"Initialization checkpoint not found: {path}")
    return path


def resolve_checkpoint_tag(cfg: dict[str, Any], cli_tag: str | None) -> str:
    tag = str(cli_tag or cfg.get("run", {}).get("ckpt_tag", "last")).lower()
    if tag not in {"best", "last"}:
        raise ValueError(f"Unsupported checkpoint tag {tag!r}; expected 'best' or 'last'")
    return tag


def resolve_folds(args: argparse.Namespace, cfg: dict[str, Any]) -> list[Fold]:
    fold_count = int(cfg["data"].get("kfold", 5))
    default_fold = parse_fold(str(cfg["data"].get("default_fold", 1)))

    if args.kfold:
        first_fold = args.start_fold if args.start_fold is not None else 1
        folds = list(range(first_fold, fold_count + 1))
    else:
        if args.start_fold is not None:
            raise ValueError("--start-fold requires --kfold")
        folds = [args.fold if args.fold is not None else default_fold]

    invalid = [
        fold
        for fold in folds
        if fold != "demo" and (not isinstance(fold, int) or fold < 1 or fold > fold_count)
    ]
    if not folds or invalid:
        raise ValueError(f"Fold indices must be within [1, {fold_count}], got {folds}")
    return folds


def print_demo_data_notice(cfg: dict[str, Any]) -> None:
    stream = sys.stderr
    use_color = (
        bool(getattr(stream, "isatty", lambda: False)())
        and "NO_COLOR" not in os.environ
    )

    def colorize(text: str, code: str) -> str:
        return f"\033[{code}m{text}\033[0m" if use_color else text

    data_cfg = cfg["data"]
    data_root = Path(data_cfg["root"]).expanduser()
    layout = str(data_cfg.get("layout", "fold_directories")).lower()
    if layout == "fold_directories":
        demo_location = data_root / "fold-demo"
        expected_layout = "<PATH>/fold-<N>/{train,valid,test}/"
    else:
        demo_location = data_root
        file_pattern = str(
            data_cfg.get(
                "file_pattern", "{dataset}_fold-{fold}_{split}_{filename}"
            )
        )
        expected_layout = f"<PATH>/{file_pattern}"

    task_name = str(cfg.get("task", {}).get("name", ""))
    if task_name in COMPLETE_PACKAGED_DEMO_TASKS:
        print(
            colorize(
                "[PACKAGED FULL DATA] fold-demo is selected. For this medicinal-herb "
                f"task, {demo_location} contains the complete packaged "
                "train/valid/test dataset.",
                "1;32",
            ),
            file=stream,
        )
        print(
            colorize(
                "[OPTIONAL 5-FOLD] To use custom five-fold partitions, place the "
                f"task files under {expected_layout} using the configured filenames, "
                "then pass --fold <1-5> or --kfold.",
                "1;34",
            ),
            file=stream,
        )
        return

    print(
        colorize(
            "[DEMO DATA] fold-demo is selected. This run uses the example dataset "
            f"in {demo_location}.",
            "1;31",
        ),
        file=stream,
    )
    print(
        colorize(
            "[FULL/CUSTOM DATA] Pass --data-root <PATH> together with --fold <1-5> or "
            "--kfold, and place the task files in the required format under "
            f"{expected_layout} "
            "using the filenames defined by the selected task config.",
            "1;34",
        ),
        file=stream,
    )


def resolve_augmentation_path(
    config_path: Path,
    cli_path: str | None,
    cfg: dict[str, Any],
) -> Path | None:
    if cli_path is not None:
        path = Path(cli_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Augmentation config not found: {path}")
        return path

    configured_path = cfg.get("augment", {}).get("path")
    if configured_path:
        path = Path(configured_path).expanduser()
        if not path.is_absolute():
            path = config_path.parent / path
        if not path.is_file():
            raise FileNotFoundError(f"Augmentation config not found: {path}")
        return path

    sibling_path = config_path.parent / "aug.yaml"
    return sibling_path if sibling_path.is_file() else None


def configure_augmentation(
    cfg: dict[str, Any],
    config_path: Path,
    cli_path: str | None,
    signal_size: int,
) -> None:
    aug_section = cfg.setdefault("augment", {})
    legacy_semantics = bool(aug_section.get("legacy_semantics", False))
    aug_section["eval_transform"] = build_eval_transform(
        signal_size, legacy_semantics=legacy_semantics
    )
    enabled = cli_path is not None or bool(
        aug_section.get("enabled", aug_section.get("path"))
    )
    aug_section["enabled"] = enabled

    if not enabled:
        aug_section["train_transform"] = build_eval_transform(
            signal_size, legacy_semantics=legacy_semantics
        )
        return

    aug_path = resolve_augmentation_path(config_path, cli_path, cfg)
    if aug_path is None:
        raise FileNotFoundError(
            "Augmentation is enabled, but no augmentation YAML was configured"
        )
    aug_section["path"] = str(aug_path)
    aug_section["train_transform"] = build_train_transform(
        load_yaml(aug_path), signal_size, legacy_semantics=legacy_semantics
    )


def configure_report_only(cfg: dict[str, Any]) -> None:
    train_cfg = cfg.setdefault("train", {})
    train_cfg.update({"save_best": False, "save_last": False, "save_every": 0})
    cfg.setdefault("run", {})["eval_in_memory_after_train"] = True
    results_cfg = cfg.setdefault("results", {})
    results_cfg.update(
        {"save_txt": False, "save_json": False, "save_predictions": False}
    )


def prepare_retrieval(task: Any, fold: Fold) -> None:
    retrieval = getattr(task, "retrieval", None)
    if not isinstance(retrieval, dict) or not retrieval.get("enabled", False):
        return

    template = retrieval.setdefault(
        "_sample_db_row_npy_template", retrieval.get("sample_db_row_npy", "")
    )
    retrieval["sample_db_row_npy"] = str(template).format_map(
        {"fold": f"fold-{fold}", "split": "test"}
    )
    mapping_path = Path(retrieval["sample_db_row_npy"])
    if not mapping_path.is_file():
        raise FileNotFoundError(f"Retrieval row mapping not found: {mapping_path}")


def prepare_eval_state(task: Any, model: Any, dataloader: DataLoader, device: str, use_bf16: bool) -> None:
    """Fit validation-only evaluation state, such as multilabel thresholds."""
    select_threshold = getattr(task, "select_eval_threshold", None)
    if not callable(select_threshold):
        return
    predictions, targets, _ = run_inference_collect(model, task, dataloader, device, use_bf16)
    select_threshold(predictions, targets)


def aggregate_fold_metrics(reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate numeric metric leaves across held-out folds.

    Non-numeric payloads such as confusion matrices and sample indices remain in
    the per-fold JSON files and are intentionally omitted from the aggregate.
    """

    def aggregate(values: list[Any]) -> Any:
        if not values:
            return None
        if all(isinstance(value, dict) for value in values):
            result = {}
            for key in sorted({key for value in values for key in value}):
                child = aggregate([value[key] for value in values if key in value])
                if child is not None:
                    result[key] = child
            return result
        if all(isinstance(value, Real) and not isinstance(value, bool) for value in values):
            numbers = [float(value) for value in values]
            return {
                "mean": float(statistics.fmean(numbers)),
                "std": float(statistics.pstdev(numbers)) if len(numbers) > 1 else 0.0,
                "n": len(numbers),
            }
        return None

    return aggregate([report["metrics"] for report in reports]) or {}


def print_fold_summary(summary: dict[str, Any]) -> None:
    print("===== K-fold Evaluation Summary =====")
    print(f"folds: {', '.join(map(str, summary['folds']))}")

    def visit(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            if set(value) >= {"mean", "std", "n"}:
                print(
                    f"{prefix}: {value['mean']:.6f} +/- {value['std']:.6f} "
                    f"(n={value['n']})"
                )
                return
            for key, child in value.items():
                visit(f"{prefix}.{key}" if prefix else str(key), child)

    visit("", summary["metrics"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        help="Task YAML config; takes precedence when --task is also provided",
    )
    parser.add_argument(
        "--task",
        help=(
            "Resolve a task config from its full name or initialism, for example "
            "bacterial_classification or bc"
        ),
    )
    parser.add_argument("--mode", choices=["train_eval", "infer_eval"])
    parser.add_argument("--fold", type=parse_fold, help="Run fold 1-5 or the demo fold")
    parser.add_argument("--kfold", action="store_true", help="Run all configured folds")
    parser.add_argument(
        "--start-fold",
        "--start_fold",
        dest="start_fold",
        type=int,
        help="First fold to run with --kfold (useful when resuming)",
    )
    parser.add_argument(
        "--ckpt",
        type=Path,
        help="Checkpoint for inference or optional initialization before training",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Require an exact state-dict match when loading --ckpt",
    )
    parser.add_argument("--device", help="Device override, for example cuda:0 or cpu")
    parser.add_argument(
        "--data-root",
        type=Path,
        help="Override config.data.root for this invocation without changing the YAML",
    )
    parser.add_argument(
        "--data-layout",
        choices=["flat", "fold_directories"],
        help="Override config.data.layout for this invocation without changing the YAML",
    )
    parser.add_argument(
        "--save-every",
        "--save_every",
        dest="save_every",
        type=int,
        help="Checkpoint interval override",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Training epoch override for this invocation without changing the YAML",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        help="DataLoader worker override for this invocation without changing the YAML",
    )
    parser.add_argument(
        "--drop-last",
        type=parse_bool,
        metavar="BOOL",
        help="Training DataLoader drop_last override, for example false",
    )
    parser.add_argument(
        "--ckpt-tag",
        "--ckpt_tag",
        dest="ckpt_tag",
        choices=["best", "last"],
    )
    parser.add_argument("--aug", help="Training augmentation YAML")
    parser.add_argument(
        "--report-only",
        "--report_only",
        dest="report_only",
        action="store_true",
        help="Print test metrics without saving checkpoints or result files",
    )
    args = parser.parse_args()
    if args.config is None and args.task is None:
        parser.error("one of --config or --task is required")
    return args


def apply_cli_overrides(cfg: dict[str, Any], args: argparse.Namespace) -> None:
    if args.mode is not None:
        cfg.setdefault("run", {})["mode"] = args.mode
    if args.device is not None:
        cfg["device"] = args.device
    if args.data_root is not None:
        cfg.setdefault("data", {})["root"] = str(args.data_root.expanduser())
    if args.data_layout is not None:
        cfg.setdefault("data", {})["layout"] = args.data_layout
    if args.save_every is not None:
        cfg.setdefault("train", {})["save_every"] = args.save_every
    if args.epochs is not None:
        if args.epochs < 1:
            raise ValueError("--epochs must be positive")
        cfg.setdefault("train", {})["epochs"] = args.epochs
    if args.num_workers is not None:
        if args.num_workers < 0:
            raise ValueError("--num-workers must be non-negative")
        cfg.setdefault("loader", {})["num_workers"] = args.num_workers
    if args.drop_last is not None:
        cfg.setdefault("loader", {})["drop_last"] = args.drop_last
    if args.ckpt_tag is not None:
        cfg.setdefault("run", {})["ckpt_tag"] = args.ckpt_tag
    if args.report_only:
        configure_report_only(cfg)


def main() -> None:
    args = parse_args()
    config_path = resolve_config_path(args.config, args.task)
    cfg = load_yaml(config_path)
    apply_cli_overrides(cfg, args)

    run_cfg = cfg.setdefault("run", {})
    mode = run_cfg.get("mode", "infer_eval")
    device = str(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    if device.startswith("cuda") and not torch.cuda.is_available():
        print(f"[run] CUDA is unavailable; using CPU instead of {device!r}")
        device = "cpu"

    distributed_requested = bool(
        run_cfg.get("ddp", False) or int(os.environ.get("WORLD_SIZE", "1")) > 1
    )
    if distributed_requested:
        raise RuntimeError(
            "Distributed downstream execution is not supported; run scripts.run "
            "as a single process."
        )

    try:
        primary_device = device
        set_seed(int(cfg.get("seed", 42)))
        if is_main_process():
            if args.config is not None and args.task is not None:
                print(
                    "[run] --config takes precedence over --task; "
                    f"config={config_path}"
                )
            elif args.task is not None:
                print(
                    f"[run] task={args.task} resolved_task={resolve_task_name(args.task)} "
                    f"config={config_path}"
                )
            else:
                print(f"[run] config={config_path}")
            print(f"[run] device={primary_device}")

        task_section = cfg.get("task", {})
        task_name = task_section.get("name")
        if not task_name:
            raise ValueError("config.task.name is required")
        task = build_task(task_name, task_section.get("args", {}) or {})
        cfg["_runtime_task_obj"] = task

        folds = resolve_folds(args, cfg)
        if "demo" in folds and is_main_process():
            print_demo_data_notice(cfg)
        model_name, model_cfg = resolve_model_config(cfg)
        method_name = str(run_cfg.get("method_name", model_name))
        if "signal_size" not in model_cfg:
            raise ValueError(f"config.model configuration for {model_name!r} needs signal_size")
        configure_augmentation(
            cfg,
            config_path,
            args.aug,
            signal_size=int(model_cfg["signal_size"]),
        )

        checkpoint_tag = resolve_checkpoint_tag(cfg, args.ckpt_tag)
        use_bf16 = bool(cfg.get("use_bf16", True))
        run_id = make_run_id()
        fold_reports: list[dict[str, Any]] = []

        for fold in folds:
            train_dl, valid_dl, test_dl, dataset_info = build_loaders_for_task(
                cfg, task, fold=fold
            )
            valid_eval_dl = valid_dl
            model = build_model(model_name, model_cfg, dataset_info)

            model = model.to(primary_device)
            checkpoint_loaded = False
            initial_checkpoint = args.ckpt
            if initial_checkpoint is None and mode == "train_eval":
                configured_checkpoint = run_cfg.get("init_ckpt")
                initial_checkpoint = Path(configured_checkpoint) if configured_checkpoint else None

            if initial_checkpoint is not None:
                initial_checkpoint = resolve_initial_checkpoint(
                    initial_checkpoint,
                    tag=str(run_cfg.get("init_ckpt_tag", "last")),
                )
                load_model_state(
                    model,
                    str(initial_checkpoint.expanduser()),
                    map_location=primary_device,
                    strict=args.strict,
                )
                checkpoint_loaded = True

            evaluation_checkpoints: list[tuple[str | None, Path | None]]
            if mode == "train_eval":
                saved_checkpoints = train_and_validate(
                    cfg=cfg,
                    model=model,
                    task=task,
                    train_dl=train_dl,
                    val_dl=valid_dl,
                    device=primary_device,
                    use_bf16=use_bf16,
                    method_name=method_name,
                    fold=fold,
                    run_id=run_id,
                )

                requested_tags = run_cfg.get("eval_ckpt_tags")
                if requested_tags is None:
                    checkpoint_value = saved_checkpoints.get(checkpoint_tag)
                    evaluation_checkpoints = [
                        (checkpoint_tag, Path(checkpoint_value) if checkpoint_value else None)
                    ]
                else:
                    evaluation_checkpoints = []
                    for tag_value in requested_tags:
                        tag = str(tag_value).lower()
                        if tag not in {"best", "last"}:
                            raise ValueError(
                                f"Unsupported run.eval_ckpt_tags entry {tag!r}"
                            )
                        checkpoint_value = saved_checkpoints.get(tag)
                        if not checkpoint_value:
                            raise FileNotFoundError(
                                f"Requested {tag!r} checkpoint was not saved"
                            )
                        evaluation_checkpoints.append((tag, Path(checkpoint_value)))

                if evaluation_checkpoints[0][1] is None and not run_cfg.get(
                    "eval_in_memory_after_train", False
                ):
                    raise FileNotFoundError(
                        f"The requested {checkpoint_tag!r} checkpoint was not saved"
                    )
            elif mode == "infer_eval":
                checkpoint_path = args.ckpt
                if checkpoint_path is None:
                    checkpoint_path = latest_checkpoint(
                        checkpoint_dir(cfg, task, method_name, fold), checkpoint_tag
                    )
                evaluation_checkpoints = [(checkpoint_tag, checkpoint_path)]
            else:
                raise ValueError(f"Unknown run mode: {mode!r}")

            requires_checkpoint = bool(getattr(task, "requires_ckpt", True))
            if mode == "infer_eval" and evaluation_checkpoints[0][1] is None and requires_checkpoint:
                raise FileNotFoundError(
                    f"No {checkpoint_tag!r} checkpoint found for fold {fold}"
                )

            if is_main_process():
                prepare_retrieval(task, fold)
                eval_model = model.module if hasattr(model, "module") else model
                for evaluation_tag, evaluation_checkpoint in evaluation_checkpoints:
                    if evaluation_checkpoint is not None and (
                        mode == "train_eval" or not checkpoint_loaded
                    ):
                        load_model_state(
                            model,
                            str(evaluation_checkpoint),
                            map_location=primary_device,
                            strict=True,
                        )
                    prepare_eval_state(
                        task, eval_model, valid_eval_dl, primary_device, use_bf16
                    )
                    output_path = infer_eval_and_save_txt(
                        cfg=cfg,
                        model=eval_model,
                        task=task,
                        dataloader=test_dl,
                        method_name=method_name,
                        fold=fold,
                        split="test",
                        device=primary_device,
                        use_bf16=use_bf16,
                        class_names=task.class_names(),
                        ckpt_path=(
                            str(evaluation_checkpoint) if evaluation_checkpoint else None
                        ),
                        tag=evaluation_tag,
                        metrics_sink=fold_reports,
                    )
                    message = (
                        f"saved: {output_path}"
                        if output_path is not None
                        else "report complete; no files saved"
                    )
                print(f"[fold-{fold}] {message}")
        if len(folds) > 1 and is_main_process():
            grouped: dict[str, list[dict[str, Any]]] = {}
            for report in fold_reports:
                tag = str(report["checkpoint_tag"])
                grouped.setdefault(tag, []).append(report)
            results_cfg = cfg.get("results", {}) or {}
            summary_root = (
                Path(results_cfg.get("root", "results"))
                / task.name
                / method_name
            )
            for tag, reports in grouped.items():
                summary = {
                    "task": task.name,
                    "method": method_name,
                    "split": "test",
                    "folds": [report["fold"] for report in reports],
                    "num_folds": len(reports),
                    "checkpoint_tag": tag,
                    "metrics": aggregate_fold_metrics(reports),
                }
                print_fold_summary(summary)
                if bool(results_cfg.get("save_json", True)):
                    ensure_dir(summary_root)
                    (summary_root / f"test_{tag}_kfold_summary.json").write_text(
                        json.dumps(summary, indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
