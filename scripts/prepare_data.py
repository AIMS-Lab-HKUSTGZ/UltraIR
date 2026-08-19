"""Prepare or validate UltraIR data from a task-oriented YAML recipe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.common.recipe import PROCESSORS, prepare_from_config, validate_from_config


TASK_NAMES = {
    "pretraining",
    "functional_group_prediction",
    "molecular_structure_elucidation",
    "physicochemical_property_prediction",
    "targeted_component_detection",
    "targeted_fractional_contribution_estimation",
    "mixture_level_component_quantification",
    "bacterial_classification",
    "medicinal_herb_geographic_origin_traceability",
    "medicinal_herb_constituent_quantification",
    "microplastics_classification",
    "soil_property_prediction",
}


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input", type=Path, help="Override prepare.input")
    parser.add_argument("--output", type=Path, help="Override prepare.output")
    parser.add_argument(
        "--workers",
        type=int,
        help="CPU workers used for independent sample conversion",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate existing prepared files without writing data",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of files owned by this preparation recipe",
    )
    return parser.parse_args()


def apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
    prepare_cfg = config.setdefault("prepare", {})
    if args.input is not None:
        prepare_cfg["input"] = str(args.input.expanduser())
    if args.output is not None:
        prepare_cfg["output"] = str(args.output.expanduser())
    if args.workers is not None:
        if args.workers < 1:
            raise ValueError("--workers must be positive")
        prepare_cfg["workers"] = args.workers
    if args.overwrite:
        prepare_cfg["overwrite"] = True


def validate_recipe(config: dict[str, Any], validate_only: bool) -> None:
    prepare_cfg = config.get("prepare")
    if not isinstance(prepare_cfg, dict):
        raise ValueError("config.prepare must be a mapping")

    task_name = prepare_cfg.get("task")
    if task_name not in TASK_NAMES:
        valid = ", ".join(sorted(TASK_NAMES))
        raise ValueError(f"config.prepare.task must be one of: {valid}")
    if not prepare_cfg.get("output"):
        raise ValueError("config.prepare.output is required")
    if prepare_cfg.get("processor") not in PROCESSORS:
        valid = ", ".join(sorted(PROCESSORS))
        raise ValueError(f"config.prepare.processor must be one of: {valid}")
    if not validate_only and not prepare_cfg.get("input"):
        raise ValueError("config.prepare.input is required when preparing data")


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = load_config(config_path)
    apply_overrides(config, args)
    validate_recipe(config, validate_only=args.validate_only)

    if args.validate_only:
        summary = validate_from_config(config)
    else:
        summary = prepare_from_config(config)
        validation = validate_from_config(config)
        summary = {"preparation": summary, "validation": validation}

    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
