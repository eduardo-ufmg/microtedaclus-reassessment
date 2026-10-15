"""What the models receive of a stream: its tuning segment and its evaluation segment.

The features of a real dataset come in units of their own, so they are
standardized, by the mean and the standard deviation that each feature has in
the tuning segment. That segment is what a deployment has before the stream
continues, so nothing of the evaluation segment enters the transformation. A
feature that is constant in the tuning segment cannot be scaled by it and is
left out. The synthetic datasets are given as they are generated.
"""

import numpy as np

from benchmark.datasets import STREAMS, load_stream
from benchmark.tuning import split


def standardize(X_tuning: np.ndarray, X_evaluation: np.ndarray):
    """Both segments scaled by the tuning segment, without its constant features."""
    deviation = X_tuning.std(axis=0)
    kept = deviation > 0
    if not kept.any():
        raise ValueError("every feature is constant in the tuning segment")
    mean = X_tuning.mean(axis=0)
    return tuple(
        np.ascontiguousarray((X[:, kept] - mean[kept]) / deviation[kept])
        for X in (X_tuning, X_evaluation)
    )


def segments(dataset: str, run: int, length: int | None = None):
    """(X, y) of the tuning segment and of the evaluation segment of a stream."""
    (X_tuning, y_tuning), (X_evaluation, y_evaluation) = split(
        *load_stream(dataset, run, length)
    )
    if STREAMS[dataset].standardized:
        X_tuning, X_evaluation = standardize(X_tuning, X_evaluation)
    return (X_tuning, y_tuning), (X_evaluation, y_evaluation)
