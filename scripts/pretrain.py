"""Pretrain the UltraIR encoder with the paper's multitask objective."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, help="Override config.data.root")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Override config.train.save_dir",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="Resume model, optimizer, and scheduler state from a checkpoint",
    )
    parser.add_argument("--device", help="Device override, for example cuda:0 or cpu")
    parser.add_argument("--epochs", type=int, help="Training epoch override")
    parser.add_argument("--batch-size", type=int, help="Batch size override")
    parser.add_argument("--num-workers", type=int, help="DataLoader worker override")
    parser.add_argument("--seed", type=int, help="Random seed override")
    return parser.parse_args()


def apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
    if args.data_root is not None:
        config.setdefault("data", {})["root"] = str(args.data_root.expanduser())
    if args.output_dir is not None:
        config.setdefault("train", {})["save_dir"] = str(args.output_dir.expanduser())
    if args.resume is not None:
        resume_path = args.resume.expanduser()
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        config.setdefault("train", {})["resume_from"] = str(resume_path)
    if args.device is not None:
        config["device"] = args.device
    if args.epochs is not None:
        if args.epochs < 1:
            raise ValueError("--epochs must be positive")
        config.setdefault("train", {})["epochs"] = args.epochs
    if args.batch_size is not None:
        if args.batch_size < 1:
            raise ValueError("--batch-size must be positive")
        config.setdefault("loader", {})["batch_size"] = args.batch_size
    if args.num_workers is not None:
        if args.num_workers < 0:
            raise ValueError("--num-workers cannot be negative")
        config.setdefault("loader", {})["num_workers"] = args.num_workers
    if args.seed is not None:
        config["seed"] = args.seed


def validate_config(config: dict[str, Any]) -> None:
    required_sections = ("data", "model", "loss", "train")
    missing = [name for name in required_sections if not isinstance(config.get(name), dict)]
    if missing:
        raise ValueError(f"Missing configuration sections: {', '.join(missing)}")

    if not config["data"].get("root"):
        raise ValueError("config.data.root is required")
    if not config["train"].get("save_dir"):
        config["train"]["save_dir"] = "checkpoints/pretraining"


def main() -> None:
    args = parse_args()
    from ultrair.pretraining.trainer import run_pretraining

    config_path = args.config.expanduser().resolve()
    config = load_config(config_path)
    apply_overrides(config, args)
    validate_config(config)

    summary = run_pretraining(config)
    if summary is not None:
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
