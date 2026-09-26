"""Declarative configuration and stage registry for iterative model corrections."""

from dataclasses import dataclass
from editorial_config import DATASET_ORDER, ITERATIVE_DATASET_ORDER


@dataclass(frozen=True)
class CorrectionStage:
    stage_id: str
    model_tag: str
    artifact_dir: str
    label_prefix: str
    models: list[str]
    datasets: list[str]


MODEL_SHORT_HEADERS = {
    "MicroTEDAClus": r"\modelname{baseline}",
    "MicroTEDAClusCorrected_orderdepend": r"\modelname{order-independent}",
    "MicroTEDAClusCorrected_dimpen": r"\modelname{no-dim-penalty}",
    "MicroTEDAClusCorrected_math": r"\modelname{corr-typicality-math}",
    "MicroTEDAClusCorrected_decay": r"\modelname{decay}",
    "MicroTEDAClusCorrected_macro": r"\modelname{macrocluster-activation}",
    "MicroTEDAClusCorrected_refined": r"\modelname{refined}",
}

STAGE_ORDER_DEPENDENCY = CorrectionStage(
    stage_id="order_dependency",
    model_tag="MicroTEDAClusCorrected_orderdepend",
    artifact_dir="artifacts_order_dependency",
    label_prefix="orderdep_",
    models=["MicroTEDAClus", "MicroTEDAClusCorrected_orderdepend"],
    datasets=DATASET_ORDER,
)

STAGE_DIMENSION_PENALTY = CorrectionStage(
    stage_id="dimension_penalty",
    model_tag="MicroTEDAClusCorrected_dimpen",
    artifact_dir="artifacts_dimension_penalty",
    label_prefix="dimpen_",
    models=[
        "MicroTEDAClus",
        "MicroTEDAClusCorrected_orderdepend",
        "MicroTEDAClusCorrected_dimpen",
    ],
    datasets=ITERATIVE_DATASET_ORDER,
)

STAGE_TYPICALITY_MATH = CorrectionStage(
    stage_id="typicality_math",
    model_tag="MicroTEDAClusCorrected_math",
    artifact_dir="artifacts_math",
    label_prefix="math_",
    models=[
        "MicroTEDAClus",
        "MicroTEDAClusCorrected_orderdepend",
        "MicroTEDAClusCorrected_dimpen",
        "MicroTEDAClusCorrected_math",
    ],
    datasets=ITERATIVE_DATASET_ORDER,
)

STAGE_DECAY = CorrectionStage(
    stage_id="decay",
    model_tag="MicroTEDAClusCorrected_decay",
    artifact_dir="artifacts_decay",
    label_prefix="decay_",
    models=[
        "MicroTEDAClus",
        "MicroTEDAClusCorrected_orderdepend",
        "MicroTEDAClusCorrected_dimpen",
        "MicroTEDAClusCorrected_math",
        "MicroTEDAClusCorrected_decay",
    ],
    datasets=ITERATIVE_DATASET_ORDER,
)

STAGE_MACROCLUSTER_ACTIVATION = CorrectionStage(
    stage_id="macrocluster_activation",
    model_tag="MicroTEDAClusCorrected_macro",
    artifact_dir="artifacts_macrocluster_activation",
    label_prefix="macro_",
    models=[
        "MicroTEDAClus",
        "MicroTEDAClusCorrected_orderdepend",
        "MicroTEDAClusCorrected_dimpen",
        "MicroTEDAClusCorrected_math",
        "MicroTEDAClusCorrected_decay",
        "MicroTEDAClusCorrected_macro",
    ],
    datasets=ITERATIVE_DATASET_ORDER,
)

STAGE_REFINED = CorrectionStage(
    stage_id="refined",
    model_tag="MicroTEDAClusCorrected_refined",
    artifact_dir="artifacts_refined",
    label_prefix="refined_",
    models=[
        "MicroTEDAClus",
        "MicroTEDAClusCorrected_orderdepend",
        "MicroTEDAClusCorrected_dimpen",
        "MicroTEDAClusCorrected_math",
        "MicroTEDAClusCorrected_decay",
        "MicroTEDAClusCorrected_macro",
        "MicroTEDAClusCorrected_refined",
    ],
    datasets=ITERATIVE_DATASET_ORDER,
)

CORRECTION_STAGES = [
    STAGE_ORDER_DEPENDENCY,
    STAGE_DIMENSION_PENALTY,
    STAGE_TYPICALITY_MATH,
    STAGE_DECAY,
    STAGE_MACROCLUSTER_ACTIVATION,
    STAGE_REFINED,
]

STAGE_BY_ID = {stage.stage_id: stage for stage in CORRECTION_STAGES}

ITERATIVE_MODEL_COLORS = {
    "MicroTEDAClus": "#1f77b4",
    "MicroTEDAClusCorrected_orderdepend": "#ff7f0e",
    "MicroTEDAClusCorrected_dimpen": "#2ca02c",
    "MicroTEDAClusCorrected_math": "#d62728",
    "MicroTEDAClusCorrected_decay": "#8c564b",
    "MicroTEDAClusCorrected_macro": "#9467bd",
    "MicroTEDAClusCorrected_refined": "#e377c2",
}

DEFAULT_RUNS = 32


def get_stage_model_datasets() -> dict[str, list[str]]:
    """Return the mapping of each model tag to the datasets it must be evaluated on.

    The baseline 'MicroTEDAClus' is evaluated across all iterative datasets.
    Each correction stage's newly introduced model is evaluated on that stage's datasets.
    """
    mapping = {
        "MicroTEDAClus": ITERATIVE_DATASET_ORDER,
    }
    for stage in CORRECTION_STAGES:
        mapping[stage.model_tag] = stage.datasets
    return mapping


def get_all_stage_tasks(runs: int = DEFAULT_RUNS) -> list[tuple[str, str, int]]:
    """Generate all canonical (dataset, model_tag, run) tasks across stages."""
    model_datasets = get_stage_model_datasets()
    tasks = []
    for model_tag, datasets in model_datasets.items():
        for dataset in datasets:
            for run in range(runs):
                tasks.append((dataset, model_tag, run))
    return tasks
