"""Models of scikit-learn that learn incrementally."""

import numpy as np
from sklearn.cluster import Birch as CoreBirch
from sklearn.cluster import MiniBatchKMeans as CoreMiniBatchKMeans

from benchmark.hyperparameters import Kind


class _Incremental:
    """A scikit-learn estimator with `partial_fit`; it predicts noise until it has learned a batch."""

    core = None

    def __init__(self):
        self._has_learned = False

    def learn(self, X):
        self.core.partial_fit(X)
        self._has_learned = True

    def predict(self, X):
        if not self._has_learned:
            return np.full(len(X), -1)
        return np.asarray(self.core.predict(X))


class MiniBatchKMeans(_Incremental):
    HYPERPARAMETERS = {"n_clusters": Kind.CLUSTERS, "reassignment_ratio": Kind.RATIO}
    SEEDED = True

    def __init__(self, n_clusters=3, reassignment_ratio=0.01, seed=0):
        super().__init__()
        self.core = CoreMiniBatchKMeans(
            n_clusters=n_clusters,
            reassignment_ratio=reassignment_ratio,
            random_state=seed,
            n_init=1,
        )

    def summary_size(self):
        return self.core.n_clusters


class Birch(_Incremental):
    HYPERPARAMETERS = {"threshold": Kind.DISTANCE, "n_clusters": Kind.CLUSTERS}

    def __init__(self, threshold=0.5, n_clusters=3):
        super().__init__()
        self.core = CoreBirch(threshold=threshold, n_clusters=n_clusters)

    def summary_size(self):
        return len(self.core.subcluster_centers_) if self._has_learned else 0
