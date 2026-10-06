"""Data streams of the experiments.

The R-generated streams (stream, streamMOA and mlbench packages) start R on first
use, so the SCL and Chameleon streams can be produced without it. `load_stream`
adds a disk cache on top of the generators, which lets parallel workers read the
R-generated streams without each of them starting R.

Usage (fills the cache; needs R):
    python -m streams FIRST_RUN LAST_RUN [DATASET ...]
"""

import logging
import sys
import urllib.request
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.datasets import make_blobs

from protocol import run_seed

CACHE_DIR = Path(__file__).resolve().parent / "data_samples_cache"

CHAMELEON_URLS = {
    "chameleon_ds3": "https://raw.githubusercontent.com/deric/clustering-benchmark/master/src/main/resources/datasets/artificial/cluto-t7-10k.arff",
    "chameleon_ds4": "https://raw.githubusercontent.com/deric/clustering-benchmark/master/src/main/resources/datasets/artificial/cluto-t8-8k.arff",
}
R_DATASETS = ("benchmark_1", "benchmark_2", "random_rbf_events", "cassini")

_robjects = None


def r_session():
    """Start R and attach the stream packages (once per process)."""
    global _robjects
    if _robjects is None:
        import rpy2.rinterface_lib.callbacks
        import rpy2.robjects as robjects
        from rpy2.robjects.packages import importr

        rpy2.rinterface_lib.callbacks.logger.setLevel(logging.ERROR)
        for package in ("stream", "streamMOA", "mlbench"):
            importr(package)
        _robjects = robjects
    return _robjects


# -----------------------------------------------------------------------------
# Deterministic Data Generation
# -----------------------------------------------------------------------------
def get_chameleon(url: str) -> tuple[np.ndarray, np.ndarray]:
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as response:
        data, _ = arff.loadarff(StringIO(response.read().decode("utf-8")))

    df = pd.DataFrame(data)
    target_col = df.columns[-1]
    y_raw = df[target_col].str.decode("utf-8").replace("noise", "-1")
    X = df.drop(columns=[target_col]).to_numpy()
    y = y_raw.astype(int).to_numpy()
    return X, y


def get_r_stream(name: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    robjects = r_session()
    robjects.r(f"set.seed({seed})")

    if name == "cassini":
        robjects.r("data <- mlbench.cassini(4000)")
        X = np.array(robjects.r("data$x"))
        y = np.nan_to_num(
            np.array(robjects.r("as.numeric(data$classes)")), nan=-1
        ).astype(int)

        rng = np.random.default_rng(seed)
        shuffle_idx = rng.permutation(len(X))
        return X[shuffle_idx], y[shuffle_idx]

    elif name == "random_rbf_events":
        robjects.r("""
            dsd <- DSD_RandomRBFGeneratorEvents(eventFrequency=1000L, eventMergeSplitOption=TRUE, eventDeleteCreate=TRUE)
            data <- get_points(dsd, n=10000, info=TRUE)
        """)
    elif name == "benchmark_1":
        robjects.r(
            "dsd <- DSD_Benchmark(1)\ndata <- get_points(dsd, n=4000, info=TRUE)"
        )
    elif name == "benchmark_2":
        robjects.r(
            "dsd <- DSD_Benchmark(2)\ndata <- get_points(dsd, n=4000, info=TRUE)"
        )
    else:
        raise ValueError(f"Unknown R dataset: {name}")

    X = np.array(robjects.r("as.matrix(data[, -ncol(data)])"))
    y = np.nan_to_num(
        np.array(robjects.r("as.numeric(data[, ncol(data)])")), nan=-1
    ).astype(int)
    return X, y


def generate_scl(dim: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X, y = make_blobs(
        n_samples=4000, centers=3, n_features=dim, cluster_std=0.5, random_state=seed
    )[0:2]
    noise = rng.uniform(low=np.min(X, axis=0), high=np.max(X, axis=0), size=(400, dim))

    X_full = np.vstack([X, noise])
    y_full = np.concatenate([y, np.full(400, -1)])

    idx = rng.permutation(len(X_full))
    return X_full[idx], y_full[idx]


def get_dataset(dataset_name: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if dataset_name.startswith("SCL-"):
        return generate_scl(int(dataset_name.split("-")[1]), seed)
    if dataset_name in CHAMELEON_URLS:
        return get_chameleon(CHAMELEON_URLS[dataset_name])
    return get_r_stream(dataset_name, seed)


# -----------------------------------------------------------------------------
# Cached access
# -----------------------------------------------------------------------------
def cache_path(dataset: str, run: int) -> Path | None:
    """Cache file of a stream, or None if the stream is cheap to regenerate."""
    if dataset.startswith("SCL-"):
        return None
    if dataset in CHAMELEON_URLS:  # does not depend on the seed
        return CACHE_DIR / f"{dataset}.npz"
    return CACHE_DIR / f"{dataset}_seed{run_seed(dataset, run)}.npz"


def load_stream(dataset: str, run: int) -> tuple[np.ndarray, np.ndarray]:
    """Stream of a (dataset, run) pair, read from the cache when it is there."""
    path = cache_path(dataset, run)
    if path is None:
        return get_dataset(dataset, run_seed(dataset, run))
    if path.exists():
        with np.load(path) as f:
            return f["X"], f["y"]
    X, y = get_dataset(dataset, run_seed(dataset, run))
    CACHE_DIR.mkdir(exist_ok=True)
    np.savez_compressed(path, X=np.asarray(X, dtype=float), y=np.asarray(y, dtype=int))
    return X, y


def missing_from_cache(datasets, runs) -> list[tuple[str, int]]:
    """(dataset, run) pairs whose stream would have to be generated."""
    missing = []
    for dataset in datasets:
        for run in runs:
            path = cache_path(dataset, run)
            if path is not None and not path.exists():
                missing.append((dataset, run))
    return missing


if __name__ == "__main__":
    first_run, last_run = int(sys.argv[1]), int(sys.argv[2])
    datasets = sys.argv[3:] or [*R_DATASETS, *CHAMELEON_URLS]
    for dataset in datasets:
        for run in range(first_run, last_run + 1):
            load_stream(dataset, run)
        print(f"{dataset}: runs {first_run}-{last_run} cached", flush=True)
