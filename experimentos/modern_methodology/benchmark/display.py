"""Names and order of datasets and models in tables and figures; their colours are
those of `pfc_shared.figures`."""

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
REAL_SHORT_NAMES = {
    "insects_abrupt": "INS-A",
    "insects_incremental": "INS-I",
    "insects_incremental_reoccurring": "INS-R",
    "gas_drift": "GAS",
    "kddcup99": "KDD",
    "covertype": "COV",
    "pendigits": "PEN",
    "shuttle": "SHU",
}

MODEL_NAMES = {
    "MicroTEDAClusPlus": "MicroTEDAClus+",
    "DenStream": "DenStream",
    "DBSTREAM": "DBSTREAM",
    "BIRCH": "BIRCH",
    "MiniBatchKMeans": "Mini-batch k-means",
    "BICO": "BICO",
    "MicroTEDAClusPlusFinal": "MicroTEDAClus+",
}


def columns(datasets) -> dict:
    """Column headers of a table with one column per dataset."""
    names = SHORT_NAMES | REAL_SHORT_NAMES
    return {dataset: names[dataset] for dataset in datasets}


def model_label(model: str) -> str:
    return rf"\modelname{{{MODEL_NAMES[model]}}}"
