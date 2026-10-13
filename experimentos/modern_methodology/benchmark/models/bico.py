"""BICO: a coreset of the stream, clustered by k-means when labels are requested."""

import numpy as np
from bico import BICO as CoreBico
from sklearn.cluster import KMeans

from benchmark.hyperparameters import Kind


class Bico:
    HYPERPARAMETERS = {"n_clusters": Kind.CLUSTERS, "summary_size": Kind.SUMMARY_SIZE}
    SEEDED = True

    def __init__(self, n_clusters=3, summary_size=600, seed=0):
        self.core = CoreBico(
            n_clusters=n_clusters,
            summary_size=summary_size,
            random_state=seed,
            coreset_estimator=KMeans(n_clusters=n_clusters, random_state=seed),
        )
        self._has_learned = False

    def learn(self, X):
        self.core.partial_fit(X)
        self._has_learned = True

    def _coreset(self):
        self.core.partial_fit()  # without samples, it computes the coreset
        return self.core.coreset_points_

    def predict(self, X):
        if not self._has_learned or len(self._coreset()) < self.core.n_clusters:
            return np.full(len(X), -1)
        return np.asarray(self.core.predict(X))

    def summary_size(self):
        return len(self._coreset()) if self._has_learned else 0
