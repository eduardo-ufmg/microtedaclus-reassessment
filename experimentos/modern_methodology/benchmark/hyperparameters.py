"""Hyperparameters by category, and the rules that give every category its search range.

A model declares the category of each hyperparameter it exposes. The range of a
category is computed from the tuning segment by one rule, the same for every model,
so no range is chosen for a model or for a dataset by hand.

The rules of the experiments, `real2`, for a tuning segment of n samples with k
labelled clusters:

- distance: a decade beyond the median distance of a sample to its nearest
  neighbour, below which a structure of that radius holds a single sample, and
  beyond the median distance between two samples, above which it holds half of the
  segment: from a tenth of the first to ten times the second;
- relative variance: the squares of the two distances above, as fractions of the
  total variance of the segment, so that on the scale of the search its range is
  that of the distances;
- horizon, a number of samples of memory or between two actions: from n / 1000
  to n, the longest horizon that the segment can tell from no forgetting;
- summary size, a number of summary structures: from 2 k to n / 4;
- minimum points, a number of samples that makes a structure dense: from 1, the
  weight of a single sample, below which the outlier threshold of DenStream cannot
  exceed one, to the square root of n, in real numbers;
- ratio: from 0.0001 to 0.999, on the logit scale, which gives every decade towards
  0 or towards 1 the same room in the search;
- multiplier, a number of standard deviations: from 1 to 10;
- clusters: from 2 to 20 k, and no more than the samples of a batch, the most
  clusters that a batch can show.

The median distance between two samples comes from a fixed sample of pairs,
because a long tuning segment has too many pairs to measure them all. The first
version of these rules, `real`, starts the ratio at 0.001; the pilot of the
real-data experiment used it for the models without a ratio, whose ranges are the
same under both.

The natural limits of the rules are the ends that no range can move beyond: the
longest horizon, n; the largest summary, the summary limit below; one minimum
point; one standard deviation; and two clusters and the samples of a batch.

No model may keep a larger summary than the largest that the summary-size rule
allows, a quarter of the tuning segment: a configuration whose summary grows beyond
it fails, in the search and in the evaluation alike. At the fine end of these
ranges a summary can hold one structure per sample, and the cost of several models
grows with the square of their summary.
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist
from scipy.special import expit, logit

from benchmark.evaluation import BATCH_SIZE

PAIRS = 1_000_000  # pairs of samples from which the rules take the median distance


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
    """Largest summary that a model may keep: the upper end of the summary-size range, a
    quarter of the tuning segment and at least twice its classes.
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


def real_ranges_of_categories(X: np.ndarray, y: np.ndarray) -> dict[Category, Range]:
    """Search range of every category by the first version of the rules, from a tuning
    segment and its labels."""
    n = len(X)
    k = max(1, len(np.unique(y[y >= 0])))
    nearest = _nearest(X) / 10
    between = 10 * _between_sampled(X)
    total_variance = float(X.var(axis=0).sum())
    return {
        Category.DISTANCE: Range(nearest, between, log=True, integer=False),
        Category.RELATIVE_VARIANCE: Range(
            nearest**2 / total_variance,
            between**2 / total_variance,
            log=True,
            integer=False,
        ),
        Category.HORIZON: _integers(n / 1000, n),
        Category.SUMMARY_SIZE: _integers(2 * k, n / 4),
        Category.MINIMUM_POINTS: Range(1.0, np.sqrt(n), log=True, integer=False),
        Category.RATIO: Range(0.001, 0.999, log=False, integer=False, logit=True),
        Category.MULTIPLIER: Range(1.0, 10.0, log=False, integer=False),
        Category.CLUSTERS: _integers(2, min(20 * k, BATCH_SIZE), log=False),
    }


def real2_ranges_of_categories(X: np.ndarray, y: np.ndarray) -> dict[Category, Range]:
    """Search range of every category by the rules of the experiments, from a tuning
    segment and its labels."""
    return real_ranges_of_categories(X, y) | {
        Category.RATIO: Range(0.0001, 0.999, log=False, integer=False, logit=True)
    }


RULES = {
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
    """Whether an end of a range of the rules is a natural limit; the upper end of the
    clusters is one where it reaches the samples of a batch."""
    if category is Category.CLUSTERS and end == "upper":
        return high >= BATCH_SIZE
    return (category, end) in NATURAL_LIMITS


def search_space(
    model_class, X: np.ndarray, y: np.ndarray, rules=real2_ranges_of_categories
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
