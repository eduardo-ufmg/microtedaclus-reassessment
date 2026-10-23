"""Canonical ordering and display configuration for generated artifacts."""

from dataclasses import dataclass

MODEL_ORDER = [
    "CluStream",
    "DenStream",
    "MicroTEDAClus",
    "CEDAS",
    "TEDACloud",
    "AutoCloud",
    "StreamKM",
    "MicroTEDAClusPlus",
]

MODEL_DISPLAY_NAMES = {"MicroTEDAClusPlus": "MicroTEDAClus+"}


def model_display_name(model_id: str) -> str:
    """Return the name of a model as it is written in tables and plots."""
    return MODEL_DISPLAY_NAMES.get(model_id, model_id)


@dataclass(frozen=True)
class DatasetSpec:
    """Stable project ID plus provenance and display aliases."""

    display_name: str
    original_paper_name: str | None = None
    source_name: str | None = None


DATASET_SPECS = {
    "benchmark_1": DatasetSpec("Stream Benchmark 1", "STR-B1", "DSD_Benchmark(1)"),
    "benchmark_2": DatasetSpec("Stream Benchmark 2", "STR-B2", "DSD_Benchmark(2)"),
    "random_rbf_events": DatasetSpec(
        "Random RBF Generator with Events",
        "RBF",
        "DSD_RandomRBFGeneratorEvents",
    ),
    "cassini": DatasetSpec("Cassini", "CSN", "mlbench.cassini"),
    "chameleon_ds3": DatasetSpec("Chameleon DS3", "ST-D1", "cluto-t7-10k.arff"),
    "chameleon_ds4": DatasetSpec("Chameleon DS4", "ST-D2", "cluto-t8-8k.arff"),
}

DATASET_ORDER = [
    "benchmark_1",
    "benchmark_2",
    "random_rbf_events",
    "cassini",
    "chameleon_ds3",
    "chameleon_ds4",
]

DATASET_DISPLAY_NAMES = {
    dataset_id: spec.display_name for dataset_id, spec in DATASET_SPECS.items()
}


def dataset_display_name(dataset_id: str) -> str:
    """Return the canonical display label for an internal dataset ID."""
    if dataset_id.startswith("SCL-"):
        return dataset_id
    return DATASET_DISPLAY_NAMES.get(dataset_id, dataset_id)


DATASET_SHORT_NAMES = {
    "benchmark_1": "B1",
    "benchmark_2": "B2",
    "random_rbf_events": "RBF",
    "cassini": "CSN",
    "chameleon_ds3": "DS3",
    "chameleon_ds4": "DS4",
}


@dataclass(frozen=True)
class Comparison:
    """A model in focus compared with each of the other models.

    Its tables and figures go to the artifacts directory of `chapter`, under
    `name`, and their labels start with `label`. `subject` names the models in
    the captions.
    """

    name: str
    focus: str
    others: tuple[str, ...]
    chapter: str
    label: str
    subject: str


COMPARISONS = [
    Comparison(
        "reassessment",
        "MicroTEDAClus",
        ("DenStream", "CluStream", "StreamKM", "AutoCloud", "CEDAS", "TEDACloud"),
        "MethodologyReassessment",
        "mr",
        "\\modelname{MicroTEDAClus} and the benchmark models",
    ),
    Comparison(
        "comparison",
        "MicroTEDAClusPlus",
        (
            "MicroTEDAClus",
            "DenStream",
            "CluStream",
            "StreamKM",
            "AutoCloud",
            "CEDAS",
            "TEDACloud",
        ),
        "ModelImprovement",
        "mi_comparison",
        "\\modelname{MicroTEDAClus+}, \\modelname{MicroTEDAClus} and the benchmark models",
    ),
]

METRIC_EXCLUDED_MODELS: dict[str, set[str]] = {
    "mc_max": {"StreamKM"},
    "mc_steadystate": {"StreamKM"},
    "microcluster_count": {"StreamKM"},
}


def effective_models(comparison: Comparison, metric: str) -> list[str]:
    """The models of a comparison, the model in focus first, minus the exclusions of a metric."""
    excluded = METRIC_EXCLUDED_MODELS.get(metric, set())
    return [m for m in (comparison.focus, *comparison.others) if m not in excluded]
