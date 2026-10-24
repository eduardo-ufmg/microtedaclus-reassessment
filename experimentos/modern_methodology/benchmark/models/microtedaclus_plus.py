"""MicroTEDAClus+, the model proposed by the thesis, with and without its noise label."""

import math

import numpy as np

from benchmark.hyperparameters import Category
from pfc_shared.microtedaclus_plus import MicroTEDAClusPlus as Core


class MicroTEDAClusPlus:
    HYPERPARAMETERS = {
        "variance_limit": Category.RELATIVE_VARIANCE,
        "decay": Category.HORIZON,
        "noise_distance": Category.MULTIPLIER,
    }

    def __init__(self, variance_limit=0.01, decay=100, noise_distance=6.0):
        self.core = Core(
            variance_limit=variance_limit, decay=decay, noise_distance=noise_distance
        )

    def learn(self, X):
        self.core.fit(X)

    def predict(self, X):
        return np.asarray(self.core.predict(X))

    def summary_size(self):
        return self.core.microcluster_count()


class MicroTEDAClusPlusNoNoise(MicroTEDAClusPlus):
    """MicroTEDAClus+ without its noise label, for data whose classes cover every sample.

    At an infinite noise distance, every sample goes to the macrocluster of its nearest
    active microcluster; before any microcluster is active, every sample is noise, as
    for every model.
    """

    HYPERPARAMETERS = {
        "variance_limit": Category.RELATIVE_VARIANCE,
        "decay": Category.HORIZON,
    }

    def __init__(self, variance_limit=0.01, decay=100):
        super().__init__(variance_limit, decay, noise_distance=math.inf)
