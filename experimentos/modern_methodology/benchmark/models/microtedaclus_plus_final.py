"""MicroTEDAClus+ in its final version, from the improvement on the real datasets."""

from dataclasses import replace

import numpy as np

from benchmark.hyperparameters import Category, Range
from pfc_shared.microtedaclus_plus_final import MicroTEDAClusPlusFinal as Core


def _variance_limit(of_category):
    """The range of the relative variance with its lower end one decade lower."""
    of = of_category[Category.RELATIVE_VARIANCE]
    return replace(of, low=of.low / 10)


class MicroTEDAClusPlusFinal:
    HYPERPARAMETERS = {
        "variance_limit": _variance_limit,
        "decay": Category.HORIZON,
        "link": Range(0.5, 4.0, log=False, integer=False),
    }

    def __init__(self, variance_limit=0.01, decay=100, link=2.0):
        self.core = Core(variance_limit=variance_limit, decay=decay, link=link)

    def learn(self, X):
        self.core.fit(X)

    def predict(self, X):
        return np.asarray(self.core.predict(X))

    def summary_size(self):
        return self.core.microcluster_count()
