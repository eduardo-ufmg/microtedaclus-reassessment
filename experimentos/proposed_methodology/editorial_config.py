"""Canonical ordering and display configuration for generated artifacts."""

from dataclasses import dataclass

MODEL_ORDER = [
    "CluStream",
    "DenStream",
    "MicroTEDAClus",
    "MicroTEDAClusCorrected",
    "CEDAS",
    "TEDACloud",
    "AutoCloud",
    "StreamKM",
]


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


SUBSET_MODEL_NAMES = {
    "all": set(MODEL_ORDER) - {"MicroTEDAClusCorrected"},
    "original-paper": {"MicroTEDAClus", "DenStream", "CluStream", "StreamKM"},
    "teda-micro-based": {"MicroTEDAClus", "AutoCloud", "CEDAS", "TEDACloud"},
    # "microtedaclus": {"MicroTEDAClus", "MicroTEDAClusCorrected"},
}

SUBSETS = {
    subset_name: [model for model in MODEL_ORDER if model in model_names]
    for subset_name, model_names in SUBSET_MODEL_NAMES.items()
}

METRIC_EXCLUDED_MODELS: dict[str, set[str]] = {
    "mc_max": {"StreamKM"},
    "mc_steadystate": {"StreamKM"},
    "microcluster_count": {"StreamKM"},
}


def effective_models(subset_name: str, metric: str) -> list[str]:
    """Return the subset's model list minus any metric-level exclusions."""
    excluded = METRIC_EXCLUDED_MODELS.get(metric, set())
    return [m for m in SUBSETS[subset_name] if m not in excluded]


MODEL_COLORS = dict(
    zip(
        MODEL_ORDER,
        [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#ffdead",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
        ],
    )
)
