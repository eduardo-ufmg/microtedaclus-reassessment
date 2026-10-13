"""MCMSTStream: microclusters in a sliding window, joined by a minimum spanning tree."""

import numpy as np
from mcmststream import MCMSTStream as Core

from benchmark.hyperparameters import Kind


class MCMSTStream:
    HYPERPARAMETERS = {
        "W": Kind.HORIZON,
        "N": Kind.MINIMUM_POINTS,
        "n_micro": Kind.MINIMUM_POINTS,
        "r": Kind.DISTANCE,
    }

    def __init__(self, W=200, N=2, n_micro=2, r=0.1):
        self.core = Core(W=W, N=N, n_micro=n_micro, r=r)

    def learn(self, X):
        self.core.partial_fit(X)

    def predict(self, X):
        labels = self.core.predict(X).astype(int)
        return np.where(labels == 0, -1, labels)  # the library labels noise 0

    def summary_size(self):
        return len(self.core.micro_clusters)
