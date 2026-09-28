"""Declarative configuration and stage registry for improved model iterations."""

from dataclasses import dataclass
from editorial_config import DATASET_ORDER, ITERATIVE_DATASET_ORDER


@dataclass(frozen=True)
class ImprovementStage:
    stage_id: str
    model_tag: str
    artifact_dir: str
    label_prefix: str
    models: list[str]
    datasets: list[str]
    control_model: str = "MicroTEDAClusCorrected_macro"


IMPROVED_MODEL_SHORT_HEADERS = {
    "MicroTEDAClus": r"\modelname{baseline}",
    "MicroTEDAClusCorrected_macro": r"\modelname{macrocluster-activation}",
    "MicroTEDAClusImproved_base": r"\modelname{base-improved}",
}

STAGE_BASE_IMPROVED = ImprovementStage(
    stage_id="base_improved",
    model_tag="MicroTEDAClusImproved_base",
    artifact_dir="artifacts_base_improved",
    label_prefix="base_imp_",
    models=[
        "MicroTEDAClus",
        "MicroTEDAClusCorrected_macro",
        "MicroTEDAClusImproved_base",
    ],
    datasets=ITERATIVE_DATASET_ORDER,
    control_model="MicroTEDAClusCorrected_macro",
)

IMPROVEMENT_STAGES = [
    STAGE_BASE_IMPROVED,
]

STAGE_BY_ID = {stage.stage_id: stage for stage in IMPROVEMENT_STAGES}

IMPROVED_MODEL_COLORS = {
    "MicroTEDAClus": "#1f77b4",
    "MicroTEDAClusCorrected_macro": "#9467bd",
    "MicroTEDAClusImproved_base": "#2ca02c",
}

DEFAULT_RUNS = 32


def get_stage_model_datasets() -> dict[str, list[str]]:
    """Return the mapping of each model tag to the datasets it must be evaluated on."""
    mapping = {
        "MicroTEDAClus": ITERATIVE_DATASET_ORDER,
        "MicroTEDAClusCorrected_macro": ITERATIVE_DATASET_ORDER,
    }
    for stage in IMPROVEMENT_STAGES:
        mapping[stage.model_tag] = stage.datasets
    return mapping


def get_all_stage_tasks(runs: int = DEFAULT_RUNS) -> list[tuple[str, str, int]]:
    """Generate all canonical (dataset, model_tag, run) tasks across improvement stages."""
    model_datasets = get_stage_model_datasets()
    tasks = []
    for model_tag, datasets in model_datasets.items():
        for dataset in datasets:
            for run in range(runs):
                tasks.append((dataset, model_tag, run))
    return tasks
