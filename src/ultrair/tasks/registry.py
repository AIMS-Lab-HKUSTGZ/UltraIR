"""Task factory exposing only the final paper tasks."""

from __future__ import annotations

from typing import Any

from .functional_group_prediction import FunctionalGroupPredictionTask
from .mixture_level_component_quantification import MixtureLevelComponentQuantificationTask
from .molecular_structure_elucidation import MolecularStructureElucidationTask
from .multiclass_classification import MulticlassClassificationTask
from .multioutput_regression import MultioutputRegressionTask
from .physicochemical_property_prediction import PhysicochemicalPropertyPredictionTask
from .targeted_component_detection import TargetedComponentDetectionTask
from .targeted_fractional_contribution_estimation import (
    TargetedFractionalContributionEstimationTask,
)


TASK_NAMES = {
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


def build_task(name: str, task_args: dict[str, Any]):
    key, args = str(name).lower(), dict(task_args or {})
    if key not in TASK_NAMES:
        raise ValueError(f"Unknown paper task {name!r}; expected one of {sorted(TASK_NAMES)}")
    if key == "functional_group_prediction":
        return FunctionalGroupPredictionTask(**args)
    if key == "molecular_structure_elucidation":
        return MolecularStructureElucidationTask(**args)
    if key == "physicochemical_property_prediction":
        return PhysicochemicalPropertyPredictionTask(**args)
    if key == "targeted_component_detection":
        return TargetedComponentDetectionTask(**args)
    if key == "targeted_fractional_contribution_estimation":
        return TargetedFractionalContributionEstimationTask(**args)
    if key == "mixture_level_component_quantification":
        return MixtureLevelComponentQuantificationTask(**args)
    if key in {
        "bacterial_classification",
        "medicinal_herb_geographic_origin_traceability",
        "microplastics_classification",
    }:
        return MulticlassClassificationTask(task_name=key, **args)
    return MultioutputRegressionTask(task_name=key, **args)
