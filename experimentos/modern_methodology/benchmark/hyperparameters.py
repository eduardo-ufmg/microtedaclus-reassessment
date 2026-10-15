"""Hyperparameters by category, and the rules that give every category its search range.

A model declares the category of each hyperparameter it exposes. The range of a
category is computed from the tuning segment by one rule, the same for every model,
so no range is chosen for a model or for a dataset by hand.

Rules, for a tuning segment of n samples with k labelled clusters:

- distance: from the median distance of a sample to its nearest neighbour,
  below which a structure of that radius holds a single sample, to the median
  distance between two samples, above which it holds half of the segment;
- relative variance: the squares of the two distances above, as fractions of
  the total variance of the segment;
- horizon, a number of samples of memory or between two actions: from n / 100
  to n, the longest horizon that the segment can tell from no forgetting;
- summary size, a number of summary structures: from 2 k to n / 4;
- minimum points, a number of samples that makes a structure dense: from 2 to
  the square root of n;
- ratio: from 0.01 to 0.99;
- multiplier, a number of standard deviations: from 1 to 10;
- clusters: from 2 to 2 k.
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist


class Category(Enum):
    DISTANCE = "distance"
    RELATIVE_VARIANCE = "relative variance"
    HORIZON = "horizon"
    SUMMARY_SIZE = "summary size"
    MINIMUM_POINTS = "minimum points"
    RATIO = "ratio"
    MULTIPLIER = "multiplier"
    CLUSTERS = "clusters"


@dataclass(frozen=True)
class Range:
    low: float
    high: float
    log: bool
    integer: bool

    def suggest(self, trial, name):
        if self.integer:
            return trial.suggest_int(name, int(self.low), int(self.high), log=self.log)
        return trial.suggest_float(name, self.low, self.high, log=self.log)


def _integers(low, high, log=True) -> Range:
    low = max(1, int(np.ceil(low)))
    return Range(low, max(low, int(np.floor(high))), log, integer=True)


def ranges_of_kinds(X: np.ndarray, y: np.ndarray) -> dict[Category, Range]:
    """Search range of every category of hyperparameter, from a tuning segment and its labels."""
    n = len(X)
    k = max(1, len(np.unique(y[y >= 0])))
    to_neighbour = cKDTree(X).query(X, k=2)[0][:, 1]
    nearest = float(np.median(to_neighbour))
    if nearest == 0.0:  # more than half of the samples are repeated
        nearest = float(to_neighbour[to_neighbour > 0].min())
    between = float(np.median(pdist(X)))
    total_variance = float(X.var(axis=0).sum())
    return {
        Category.DISTANCE: Range(nearest, between, log=True, integer=False),
        Category.RELATIVE_VARIANCE: Range(
            nearest**2 / total_variance,
            between**2 / total_variance,
            log=True,
            integer=False,
        ),
        Category.HORIZON: _integers(n / 100, n),
        Category.SUMMARY_SIZE: _integers(2 * k, n / 4),
        Category.MINIMUM_POINTS: _integers(2, np.sqrt(n)),
        Category.RATIO: Range(0.01, 0.99, log=False, integer=False),
        Category.MULTIPLIER: Range(1.0, 10.0, log=False, integer=False),
        Category.CLUSTERS: _integers(2, 2 * k, log=False),
    }


def search_space(model_class, X: np.ndarray, y: np.ndarray) -> dict[str, Range]:
    """Search range of every hyperparameter of a model, from a tuning segment and its labels."""
    of_kind = ranges_of_kinds(X, y)
    return {
        name: of_kind[category]
        for name, category in model_class.HYPERPARAMETERS.items()
    }
