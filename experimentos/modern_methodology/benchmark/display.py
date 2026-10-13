"""Names, order and colours of datasets and models in tables and figures."""

from benchmark.datasets import SCALABILITY_DIMENSIONS

SHORT_NAMES = {
    "benchmark_1": "B1",
    "benchmark_2": "B2",
    "random_rbf_events": "RBF",
    "cassini": "CSN",
    "chameleon_ds3": "DS3",
    "chameleon_ds4": "DS4",
    **{f"SCL-{d}": f"S{d}" for d in SCALABILITY_DIMENSIONS},
}
SCALABILITY_DATASETS = [f"SCL-{d}" for d in SCALABILITY_DIMENSIONS]
MAIN_DATASETS = [d for d in SHORT_NAMES if d not in SCALABILITY_DATASETS]

MODEL_NAMES = {
    "MicroTEDAClusPlus": "MicroTEDAClus+",
    "CluStream": "CluStream",
    "DenStream": "DenStream",
    "DBSTREAM": "DBSTREAM",
    "BIRCH": "BIRCH",
    "MiniBatchKMeans": "Mini-batch k-means",
    "BICO": "BICO",
    "MCMSTStream": "MCMSTStream",
}
MODEL_COLORS = {
    "MicroTEDAClusPlus": "xkcd:red",
    "CluStream": "xkcd:blue",
    "DenStream": "xkcd:green",
    "DBSTREAM": "xkcd:orange",
    "BIRCH": "xkcd:purple",
    "MiniBatchKMeans": "xkcd:brown",
    "BICO": "xkcd:teal",
    "MCMSTStream": "xkcd:grey",
}


def columns(datasets) -> dict:
    """Column headers of a table with one column per dataset."""
    return {dataset: SHORT_NAMES[dataset] for dataset in datasets}


def model_label(model: str) -> str:
    return rf"\modelname{{{MODEL_NAMES[model]}}}"
