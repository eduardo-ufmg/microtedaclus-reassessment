"""Streams of the modernized experiment.

A stream is identified by a dataset and a run. Its random generator is seeded
by the base seed, the dataset and the run, so the same pair gives the same
stream on any machine. The synthetic datasets are those of `DATASETS`; the real
datasets, read from files by `benchmark.real_data`, are those of
`REAL_DATASETS`, and their features are standardized by the tuning segment of
each stream, see `benchmark.preprocessing`.
"""

import zlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.io import arff

from benchmark.generators.cassini import cassini
from benchmark.generators.moving_gaussians import benchmark_1, benchmark_2
from benchmark.generators.rbf_events import rbf_events
from benchmark import real_data

BASE_SEED = 42
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SCALABILITY_DIMENSIONS = (1, 2, 4, 8, 16, 32, 64)


@dataclass(frozen=True)
class Dataset:
    generate: Callable  # (length, rng) -> (X, y), with noise labelled -1
    length: int | None  # number of samples; None for a file of fixed size
    shuffle: bool  # whether the order of the samples is random
    standardized: bool = (
        False  # whether the features are standardized by the tuning segment
    )


def _chameleon(file_name):
    @lru_cache(maxsize=None)
    def read():
        data, _ = arff.loadarff(DATA_DIR / file_name)
        X = np.column_stack([data[name] for name in data.dtype.names[:-1]]).astype(
            float
        )
        labels = np.char.decode(data[data.dtype.names[-1]])
        return X, np.where(labels == "noise", "-1", labels).astype(int)

    return lambda length, rng: read()


def _scalability(dimension):
    def generate(length, rng):
        """Three Gaussian clusters in a box, with one sample of uniform noise for every ten."""
        n_clustered = round(length / 1.1)
        centers = rng.uniform(-10, 10, size=(3, dimension))
        y = np.arange(n_clustered) % 3
        X = centers[y] + rng.normal(0.0, 0.5, size=(n_clustered, dimension))
        noise = rng.uniform(
            X.min(axis=0), X.max(axis=0), size=(length - n_clustered, dimension)
        )
        return np.vstack([X, noise]), np.concatenate([y, np.full(len(noise), -1)])

    return generate


DATASETS = {
    "benchmark_1": Dataset(benchmark_1, 4000, shuffle=False),
    "benchmark_2": Dataset(benchmark_2, 4000, shuffle=False),
    "random_rbf_events": Dataset(rbf_events, 10000, shuffle=False),
    "cassini": Dataset(cassini, 4000, shuffle=True),
    "chameleon_ds3": Dataset(_chameleon("cluto-t7-10k.arff"), None, shuffle=True),
    "chameleon_ds4": Dataset(_chameleon("cluto-t8-8k.arff"), None, shuffle=True),
    **{
        f"SCL-{dimension}": Dataset(_scalability(dimension), 4400, shuffle=True)
        for dimension in SCALABILITY_DIMENSIONS
    },
}


def _gas_drift(length, rng):
    """The batches in their order, which carries the drift, each shuffled: its order is not recorded."""
    X, y, batches = real_data.gas_drift()
    order = np.concatenate(
        [
            rng.permutation(np.flatnonzero(batches == batch))
            for batch in np.unique(batches)
        ]
    )
    return X[order], y[order]


def _file(reader, *arguments):
    return lambda length, rng: reader(*arguments)


REAL_DATASETS = {
    "insects_abrupt": Dataset(
        _file(real_data.insects, "insects_abrupt/abrupt_balanced.csv"),
        None,
        shuffle=False,
        standardized=True,
    ),
    "insects_incremental": Dataset(
        _file(real_data.insects, "insects_incremental/incremental_balanced.csv"),
        None,
        shuffle=False,
        standardized=True,
    ),
    "insects_incremental_reoccurring": Dataset(
        _file(
            real_data.insects,
            "insects_incremental_reoccurring/incremental_reoccurring_balanced.csv",
        ),
        None,
        shuffle=False,
        standardized=True,
    ),
    "gas_drift": Dataset(_gas_drift, None, shuffle=False, standardized=True),
    "kddcup99": Dataset(
        _file(real_data.kddcup99), None, shuffle=False, standardized=True
    ),
    "covertype": Dataset(
        _file(real_data.covertype), None, shuffle=False, standardized=True
    ),
    "pendigits": Dataset(
        _file(real_data.pendigits), None, shuffle=True, standardized=True
    ),
    "shuttle": Dataset(_file(real_data.shuttle), None, shuffle=True, standardized=True),
}
STREAMS = {**DATASETS, **REAL_DATASETS}


def stream_rng(dataset: str, run: int) -> np.random.Generator:
    return np.random.default_rng([BASE_SEED, zlib.crc32(dataset.encode()), run])


def load_stream(dataset: str, run: int, length: int | None = None):
    """Samples and labels of a stream; `length` overrides the length of a generated dataset."""
    spec = STREAMS[dataset]
    rng = stream_rng(dataset, run)
    X, y = spec.generate(length or spec.length, rng)
    if spec.shuffle:
        order = rng.permutation(len(X))
        X, y = X[order], y[order]
    return np.ascontiguousarray(X, dtype=float), np.asarray(y, dtype=int)
