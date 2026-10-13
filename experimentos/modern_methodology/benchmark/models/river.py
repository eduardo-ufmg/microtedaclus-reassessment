"""Models of River, which learn and predict one sample at a time.

River measures time in samples. A rate of forgetting is exposed here as its
half-life, the number of samples after which a weight halves.
"""

import numpy as np
from river import cluster

from benchmark.evaluation import BATCH_SIZE
from benchmark.hyperparameters import Kind


def _samples(X):
    """Samples as River takes them: one dictionary from feature index to value per sample."""
    return [dict(enumerate(row)) for row in X.tolist()]


def _noise(X):
    return np.full(len(X), -1)


class CluStream:
    HYPERPARAMETERS = {
        "n_macro_clusters": Kind.CLUSTERS,
        "max_micro_clusters": Kind.SUMMARY_SIZE,
        "micro_cluster_r_factor": Kind.MULTIPLIER,
        "time_window": Kind.HORIZON,
    }
    SEEDED = True

    def __init__(
        self,
        n_macro_clusters=5,
        max_micro_clusters=100,
        micro_cluster_r_factor=2.0,
        time_window=1000,
        seed=0,
    ):
        # The macroclusters are computed once for every batch that the protocol predicts.
        self.core = cluster.CluStream(
            n_macro_clusters=n_macro_clusters,
            max_micro_clusters=max_micro_clusters,
            micro_cluster_r_factor=micro_cluster_r_factor,
            time_window=time_window,
            time_gap=BATCH_SIZE,
            seed=seed,
        )

    def learn(self, X):
        for x in _samples(X):
            self.core.learn_one(x)

    def predict(self, X):
        if not self.core.centers:
            return _noise(X)
        return np.array([self.core.predict_one(x) for x in _samples(X)])

    def summary_size(self):
        return len(self.core.micro_clusters)


class DBSTREAM:
    HYPERPARAMETERS = {
        "clustering_threshold": Kind.DISTANCE,
        "half_life": Kind.HORIZON,
        "cleanup_interval": Kind.HORIZON,
        "intersection_factor": Kind.RATIO,
        "minimum_weight": Kind.MINIMUM_POINTS,
    }

    def __init__(
        self,
        clustering_threshold=1.0,
        half_life=100,
        cleanup_interval=2,
        intersection_factor=0.3,
        minimum_weight=1.0,
    ):
        self.core = cluster.DBSTREAM(
            clustering_threshold=clustering_threshold,
            fading_factor=1.0 / half_life,
            cleanup_interval=cleanup_interval,
            intersection_factor=intersection_factor,
            minimum_weight=minimum_weight,
        )

    def learn(self, X):
        for x in _samples(X):
            self.core.learn_one(x)

    def predict(self, X):
        if self.core.n_clusters == 0:
            return _noise(X)
        return np.array([self.core.predict_one(x) for x in _samples(X)])

    def summary_size(self):
        return len(self.core.micro_clusters)


class DenStream:
    """DenStream; `outlier_ratio` places the outlier threshold beta * mu between 1 and mu."""

    HYPERPARAMETERS = {
        "epsilon": Kind.DISTANCE,
        "half_life": Kind.HORIZON,
        "mu": Kind.MINIMUM_POINTS,
        "outlier_ratio": Kind.RATIO,
        "n_samples_init": Kind.HORIZON,
    }

    def __init__(
        self, epsilon=0.02, half_life=100, mu=2, outlier_ratio=0.5, n_samples_init=100
    ):
        self.core = cluster.DenStream(
            decaying_factor=1.0 / half_life,
            beta=(1.0 + outlier_ratio * (mu - 1.0)) / mu,
            mu=mu,
            epsilon=epsilon,
            n_samples_init=n_samples_init,
            stream_speed=1,
        )

    def learn(self, X):
        for x in _samples(X):
            self.core.learn_one(x)

    def predict(self, X):
        """Labels of a batch, with the offline clustering computed once for the batch.

        River computes the clusters again at every predicted sample, although
        they only change when the model learns. The first sample of the batch
        is predicted by River, and the others are assigned to the clusters that
        this prediction left, by the function that River itself uses.
        """
        if not self.core.initialized:
            return _noise(X)
        first, *others = _samples(X)
        labels = [self.core.predict_one(first)]
        clusters = self.core.clusters
        labels += [self.core._get_closest_cluster_key(x, clusters) for x in others]
        return np.array(labels)

    def summary_size(self):
        return len(self.core.p_micro_clusters) + len(self.core.o_micro_clusters)
