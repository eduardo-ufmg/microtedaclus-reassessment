"""MicroTEDAClus+, the model proposed by the thesis."""

import numpy as np

from benchmark.hyperparameters import Kind
from pfc_shared.microtedaclus_plus import MicroTEDAClusPlus as Core


class MicroTEDAClusPlus:
    HYPERPARAMETERS = {
        "variance_limit": Kind.RELATIVE_VARIANCE,
        "decay": Kind.HORIZON,
        "noise_distance": Kind.MULTIPLIER,
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
