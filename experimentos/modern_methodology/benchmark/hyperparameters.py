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

The widened rules, for the real datasets, move every end of a range that the
synthetic experiment found binding one decade further, in the scale of the
search; the minimum points and the ratio are the two exceptions:

- distance: from a tenth of the nearest-neighbour distance to ten times the
  distance between two samples;
- relative variance: its upper end ten times higher;
- horizon: from n / 1000;
- minimum points: from 1, the weight of a single sample, below which the outlier
  threshold of DenStream cannot exceed one, and in real numbers;
- ratio: from 0.001 to 0.999, on the logit scale, which gives every decade
  towards 0 or towards 1 the same room in the search;
- clusters: up to 4 k.

They also take the median distance between two samples from a fixed sample of
pairs, because a long tuning segment has too many pairs to measure them all.

The real-data rules are the widened rules with two corrections:

- relative variance: the squares of the widened distances, as fractions of the
  total variance, so that on the scale of the search its range is that of the
  distances;
- clusters: up to 20 k, ten times the upper end of the synthetic rules, and no
  more than the samples of a batch, the most clusters that a batch can show.

Their second version, `real2`, also moves the lower end of the ratio one decade
further, to 0.0001, because the pilot of the first found the reassignment ratio of
Mini-batch k-means at 0.001 in a quarter of the runs of two datasets.

The natural limits of the real-data rules are the ends that no widening moves:
the longest horizon, n; the largest summary, the summary limit below; one
minimum point; one standard deviation; and two clusters and the samples of a
batch.

Under the widened and both real-data rules, no model may keep a larger summary than
the largest that the summary-size rule allows, a quarter of the tuning segment: a
configuration whose summary grows beyond it fails, in the search and in the
evaluation alike. At the fine end of these ranges a summary can hold one structure
per sample, and the cost of several models grows with the square of their summary.
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist
from scipy.special import expit, logit

from benchmark.evaluation import BATCH_SIZE

PAIRS = (
    1_000_000  # pairs of samples from which the widened rules take the median distance
)


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
    logit: bool = False  # searched on the logit scale; for ranges within (0, 1)

    def suggest(self, trial, name):
        if self.integer:
            return trial.suggest_int(name, int(self.low), int(self.high), log=self.log)
        if self.logit:
            return float(
                expit(trial.suggest_float(name, logit(self.low), logit(self.high)))
            )
        return trial.suggest_float(name, self.low, self.high, log=self.log)


def summary_limit(X: np.ndarray, y: np.ndarray) -> int:
    """Largest summary that a model may keep under the widened rules: the upper end of the
    summary-size range, a quarter of the tuning segment and at least twice its classes.
    """
    k = max(1, len(np.unique(y[y >= 0])))
    return max(2 * k, len(X) // 4)


def _integers(low, high, log=True) -> Range:
    low = max(1, int(np.ceil(low)))
    return Range(low, max(low, int(np.floor(high))), log, integer=True)


def _nearest(X: np.ndarray) -> float:
    """Median distance from a sample to its nearest neighbour."""
    to_neighbour = cKDTree(X).query(X, k=2)[0][:, 1]
    nearest = float(np.median(to_neighbour))
    if nearest == 0.0:  # more than half of the samples are repeated
        nearest = float(to_neighbour[to_neighbour > 0].min())
    return nearest


def _between_sampled(X: np.ndarray, pairs=PAIRS, chunk=50_000) -> float:
    """Median distance between two samples, over a fixed sample of pairs if there are more."""
    n = len(X)
    if n * (n - 1) // 2 <= pairs:
        return float(np.median(pdist(X)))
    rng = np.random.default_rng(0)
    first = rng.integers(n, size=pairs)
    second = rng.integers(n - 1, size=pairs)
    second += second >= first  # a pair of two different samples
    distances = np.concatenate(
        [
            np.linalg.norm(
                X[first[at : at + chunk]] - X[second[at : at + chunk]], axis=1
            )
            for at in range(0, pairs, chunk)
        ]
    )
    return float(np.median(distances))


def ranges_of_categories(X: np.ndarray, y: np.ndarray) -> dict[Category, Range]:
    """Search range of every category of hyperparameter, from a tuning segment and its labels."""
    n = len(X)
    k = max(1, len(np.unique(y[y >= 0])))
    nearest = _nearest(X)
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


def widened_ranges_of_categories(X: np.ndarray, y: np.ndarray) -> dict[Category, Range]:
    """Search range of every category by the widened rules, for the real datasets."""
    n = len(X)
    k = max(1, len(np.unique(y[y >= 0])))
    nearest = _nearest(X)
    between = _between_sampled(X)
    total_variance = float(X.var(axis=0).sum())
    return {
        Category.DISTANCE: Range(nearest / 10, 10 * between, log=True, integer=False),
        Category.RELATIVE_VARIANCE: Range(
            nearest**2 / total_variance,
            10 * between**2 / total_variance,
            log=True,
            integer=False,
        ),
        Category.HORIZON: _integers(n / 1000, n),
        Category.SUMMARY_SIZE: _integers(2 * k, n / 4),
        Category.MINIMUM_POINTS: Range(1.0, np.sqrt(n), log=True, integer=False),
        Category.RATIO: Range(0.001, 0.999, log=False, integer=False, logit=True),
        Category.MULTIPLIER: Range(1.0, 10.0, log=False, integer=False),
        Category.CLUSTERS: _integers(2, 4 * k, log=False),
    }


def real_ranges_of_categories(X: np.ndarray, y: np.ndarray) -> dict[Category, Range]:
    """Search range of every category by the real-data rules."""
    of_category = widened_ranges_of_categories(X, y)
    distance = of_category[Category.DISTANCE]
    k = max(1, len(np.unique(y[y >= 0])))
    total_variance = float(X.var(axis=0).sum())
    return of_category | {
        Category.RELATIVE_VARIANCE: Range(
            distance.low**2 / total_variance,
            distance.high**2 / total_variance,
            log=True,
            integer=False,
        ),
        Category.CLUSTERS: _integers(2, min(20 * k, BATCH_SIZE), log=False),
    }


def real2_ranges_of_categories(X: np.ndarray, y: np.ndarray) -> dict[Category, Range]:
    """Search range of every category by the second version of the real-data rules."""
    return real_ranges_of_categories(X, y) | {
        Category.RATIO: Range(0.0001, 0.999, log=False, integer=False, logit=True)
    }


RULES = {
    "rules": ranges_of_categories,
    "widened": widened_ranges_of_categories,
    "real": real_ranges_of_categories,
    "real2": real2_ranges_of_categories,
}

NATURAL_LIMITS = {
    (Category.HORIZON, "upper"),
    (Category.SUMMARY_SIZE, "upper"),
    (Category.MINIMUM_POINTS, "lower"),
    (Category.MULTIPLIER, "lower"),
    (Category.CLUSTERS, "lower"),
}


def natural_limit(category: Category, end: str, high: float) -> bool:
    """Whether an end of a range of the real-data rules is a natural limit; the upper end
    of the clusters is one where it reaches the samples of a batch."""
    if category is Category.CLUSTERS and end == "upper":
        return high >= BATCH_SIZE
    return (category, end) in NATURAL_LIMITS


def search_space(
    model_class, X: np.ndarray, y: np.ndarray, rules=ranges_of_categories
) -> dict[str, Range]:
    """Search range of every hyperparameter of a model, from a tuning segment and its labels.

    A model declares, for each hyperparameter, its category, a fixed Range, or a
    function that derives its Range from the ranges of the categories.
    """
    of_category = rules(X, y)
    space = {}
    for name, spec in model_class.HYPERPARAMETERS.items():
        if isinstance(spec, Category):
            space[name] = of_category[spec]
        elif isinstance(spec, Range):
            space[name] = spec
        else:
            space[name] = spec(of_category)
    return space
