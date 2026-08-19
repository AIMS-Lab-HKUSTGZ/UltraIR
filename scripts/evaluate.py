"""Evaluate an UltraIR checkpoint using the canonical experiment runner."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

if __package__:
    from .run import parse_fold
else:
    from run import parse_fold


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
    folds = parser.add_mutually_exclusive_group()
    folds.add_argument("--fold", type=parse_fold, help="Evaluate fold 1-5 or the demo fold")
    folds.add_argument("--kfold", action="store_true", help="Evaluate all configured folds")
    parser.add_argument(
        "--start-fold",
        "--start_fold",
        dest="start_fold",
        type=int,
        help="First fold to evaluate with --kfold",
    )
    parser.add_argument(
        "--ckpt",
        type=Path,
        help="Explicit checkpoint; otherwise resolve the latest checkpoint per fold",
    )
    parser.add_argument(
        "--ckpt-tag",
        "--ckpt_tag",
        dest="ckpt_tag",
        choices=["best", "last"],
    )
    parser.add_argument("--device", help="Device override, for example cuda:0 or cpu")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Require an exact state-dict match",
    )
    parser.add_argument(
        "--report-only",
        "--report_only",
        dest="report_only",
        action="store_true",
        help="Print metrics without saving result files",
    )
    args = parser.parse_args()
    if args.config is None and args.task is None:
        parser.error("one of --config or --task is required")
    return args


def build_run_command(args: argparse.Namespace) -> list[str]:
    run_script = Path(__file__).with_name("run.py")
    command = [
        sys.executable,
        str(run_script),
    ]
    if args.config is not None:
        command.extend(["--config", str(args.config.expanduser())])
    else:
        command.extend(["--task", args.task])
    command.extend(["--mode", "infer_eval"])

    if args.fold is not None:
        command.extend(["--fold", str(args.fold)])
    if args.kfold:
        command.append("--kfold")
    if args.start_fold is not None:
        if not args.kfold:
            raise ValueError("--start-fold requires --kfold")
        command.extend(["--start-fold", str(args.start_fold)])
    if args.ckpt is not None:
        if args.kfold:
            raise ValueError(
                "A single --ckpt cannot be used with --kfold; omit it to resolve "
                "each fold from the configured checkpoint directory"
            )
        checkpoint = args.ckpt.expanduser()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        command.extend(["--ckpt", str(checkpoint)])
    if args.ckpt_tag is not None:
        command.extend(["--ckpt-tag", args.ckpt_tag])
    if args.device is not None:
        command.extend(["--device", args.device])
    if args.strict:
        command.append("--strict")
    if args.report_only:
        command.append("--report-only")
    return command


def main() -> None:
    args = parse_args()
    completed = subprocess.run(build_run_command(args), check=False)
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
