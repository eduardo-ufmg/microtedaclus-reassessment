from __future__ import annotations

import math
from dataclasses import dataclass, field

import networkx as nx
import numpy as np
from numba import njit
from scipy.spatial.distance import cdist

EPSILON = 1e-12
PROFILE_MODULE_PATH = __file__


@njit(cache=True)
def _jit_compute_new_mean(mean, x, num_samples):
    n_plus_1 = num_samples + 1
    return (num_samples / n_plus_1) * mean + x / n_plus_1


@njit(cache=True)
def _jit_compute_new_variance(variance, x, new_mean, num_samples):
    if num_samples <= 0 or len(x) == 0:
        return 0.0
    diff = x - new_mean
    sq_norm = np.dot(diff, diff)
    n_plus_1 = num_samples + 1
    return (num_samples / n_plus_1) * variance + (1 / num_samples) * (
        (np.sqrt(sq_norm) * 2 / len(x)) ** 2
    )


@njit(cache=True)
def _jit_compute_eccentricity(x, new_mean, new_variance, num_samples, variance_limit):
    n_plus_1 = num_samples + 1
    if new_variance > variance_limit and num_samples <= 3:
        return 1e3
    if new_variance <= EPSILON:
        if n_plus_1 > 1:
            return 1 / n_plus_1
        return 0.0
    diff = new_mean - x
    return 1 / n_plus_1 + np.dot(diff, diff) / (n_plus_1 * new_variance)


@njit(cache=True)
def _jit_has_intersection(mean1, mean2, var1, var2):
    diff = mean1 - mean2
    dist = np.sqrt(np.dot(diff, diff))
    threshold = 2 * (np.sqrt(var1) + np.sqrt(var2))
    return dist < threshold


@dataclass
class MicroClusterClean:
    id: int
    mean: np.ndarray
    variance_limit: float = 0.01
    num_samples: int = 1
    variance: float = 0
    life: float = 1
    decay: float = 1000
    update: bool = True
    last_eccentricity: float | None = None
    last_normalized_eccentricity: float | None = None
    last_typicality: float | None = None
    last_normalized_typicality: float | None = None
    last_density: float | None = None

    def compute_new_mean(self, x: np.ndarray):
        return _jit_compute_new_mean(self.mean, x, self.num_samples)

    def compute_new_variance(self, x: np.ndarray, new_mean: np.ndarray):
        return _jit_compute_new_variance(self.variance, x, new_mean, self.num_samples)

    def compute_eccentricity(self, x, new_mean: np.ndarray, new_variance: float):
        return _jit_compute_eccentricity(
            x, new_mean, new_variance, self.num_samples, self.variance_limit
        )

    def compute_normalized_eccentricity(self):
        if self.last_eccentricity is None:
            return None
        return self.last_eccentricity / 2

    def compute_last_typicality(self):
        if self.last_eccentricity is None:
            return None
        return 1 - self.last_eccentricity if (self.num_samples + 1) >= 2 else 0

    def compute_normalized_typicality(self):
        if self.last_typicality is None:
            return None
        return (
            self.last_typicality / (self.num_samples + 1 - 2)
            if (self.num_samples + 1) > 2
            else 0
        )

    def compute_mik_sik(self, s_ik):
        return 3 / (1 + math.exp(-0.007 * (s_ik - 100)))

    def is_outlier(self, new_variance: float):
        s_ik = self.num_samples + 1

        if s_ik < 3:
            return new_variance > self.variance_limit

        if self.last_normalized_eccentricity is None:
            return False

        mik_sik = self.compute_mik_sik(s_ik)
        outlier_limit = ((mik_sik**2) + 1) / (2 * s_ik)
        return self.last_normalized_eccentricity > outlier_limit

    def _is_outlier(self, new_variance: float):
        return self.is_outlier(new_variance)

    def decline_life_linear(self):
        if not self.decay:
            return self.life
        return self.life - 1 / self.decay

    def _update_microcluster(
        self, _is_outlier: bool, new_mean: np.ndarray, new_variance: float
    ):
        self.num_samples = self.num_samples + 1 if not _is_outlier else self.num_samples
        self.mean = new_mean if not _is_outlier else self.mean
        self.variance = new_variance if not _is_outlier else self.variance
        self.life = 1 if not _is_outlier else self.decline_life_linear()
        self.update = not _is_outlier

    def update_microcluster(self, x: np.ndarray, update=True):
        new_mean = self.compute_new_mean(x)
        new_variance = float(self.compute_new_variance(x, new_mean))
        self.last_eccentricity = self.compute_eccentricity(x, new_mean, new_variance)
        self.last_normalized_eccentricity = self.compute_normalized_eccentricity()
        self.last_typicality = self.compute_last_typicality()
        self.last_normalized_typicality = self.compute_normalized_typicality()
        if not self.last_normalized_eccentricity:
            self.last_density = None
        else:
            self.last_density = 1 / self.last_normalized_eccentricity
        is_outlier = self._is_outlier(new_variance)
        if update:
            self._update_microcluster(is_outlier, new_mean, new_variance)

    def has_intersection(self, mi):
        return _jit_has_intersection(self.mean, mi.mean, self.variance, mi.variance)


def _clone_microcluster_for_prediction(mi: MicroClusterClean) -> MicroClusterClean:
    clone = MicroClusterClean(
        id=mi.id,
        mean=mi.mean.copy(),
        variance_limit=mi.variance_limit,
        num_samples=mi.num_samples,
        variance=mi.variance,
        life=mi.life,
        decay=mi.decay,
        update=mi.update,
    )
    clone.last_eccentricity = mi.last_eccentricity
    clone.last_normalized_eccentricity = mi.last_normalized_eccentricity
    clone.last_typicality = mi.last_typicality
    clone.last_normalized_typicality = mi.last_normalized_typicality
    clone.last_density = mi.last_density
    return clone


@dataclass
class EvolvingMicroTEDAClean:
    microclusters: list[MicroClusterClean] = field(default_factory=list)
    microclusters_by_id: dict[int, MicroClusterClean] = field(default_factory=dict)
    macroclusters: list[list[int]] = field(default_factory=list)
    variance_limit: float = 0.01
    decay: float = 1000
    graph: nx.Graph = field(default_factory=nx.Graph)
    counter_microclusters: int = 0

    def create_microcluster(self, x: np.ndarray):
        new_microcluster = MicroClusterClean(
            id=self.counter_microclusters,
            mean=x,
            variance_limit=self.variance_limit,
            decay=self.decay,
        )

        self.microclusters.append(new_microcluster)
        self.microclusters_by_id[new_microcluster.id] = new_microcluster

        self.counter_microclusters += 1

    def remove_microcluster(self):
        delete_microclusters_ids = [mi.id for mi in self.microclusters if mi.life <= 0]
        delete_microclusters_ids_set = set(delete_microclusters_ids)

        for mi_id in delete_microclusters_ids:
            if mi_id in self.graph.nodes:
                self.graph.remove_node(mi_id)
            self.microclusters_by_id.pop(mi_id, None)
        self.microclusters = [
            mi for mi in self.microclusters if mi.id not in delete_microclusters_ids_set
        ]

    def update_microclusters(self, x: np.ndarray):

        if not self.microclusters:
            self.create_microcluster(x)
        else:
            for mi in self.microclusters:
                mi.update_microcluster(x)

            if not any(mi.update for mi in self.microclusters):
                self.create_microcluster(x)

        self.remove_microcluster()

    def get_microcluster(self, _id):
        return self.microclusters_by_id.get(_id)

    def disable_microclusters(self, microclusters_ids):
        microclusters = [self.get_microcluster(mi) for mi in microclusters_ids]
        microclusters = [mi for mi in microclusters if mi is not None]
        densities = [
            mi.last_density for mi in microclusters if mi.last_density is not None
        ]
        if not densities:
            return []

        mean_density = float(np.mean(densities))
        delete_microclusters_ids = [
            mi.id
            for mi in microclusters
            if mi.last_density is not None and mi.last_density < mean_density
        ]
        delete_microclusters_ids_set = set(delete_microclusters_ids)

        for mi_id in delete_microclusters_ids:
            if mi_id in self.graph.nodes:
                self.graph.remove_node(mi_id)

        return [
            mi_id
            for mi_id in [mi.id for mi in microclusters]
            if mi_id not in delete_microclusters_ids_set
        ]

    def update_macroclusters(self):
        count = len(self.microclusters)
        if count == 0:
            self.macroclusters = []
            return

        means = np.array([mi.mean for mi in self.microclusters])
        variances = np.array([mi.variance for mi in self.microclusters])
        ids = [mi.id for mi in self.microclusters]

        if count > 1:
            distances = cdist(means, means, metric="euclidean")
            for i in range(count):
                for j in range(i + 1, count):
                    threshold = 2 * (np.sqrt(variances[i]) + np.sqrt(variances[j]))
                    edge = (ids[i], ids[j])
                    if distances[i, j] < threshold:
                        self.graph.add_edge(*edge)
                    elif ids[j] in self.graph.nodes and (
                        ids[i] in self.graph.neighbors(ids[j])
                    ):
                        self.graph.remove_edge(*edge)

        components = list(nx.connected_components(self.graph))
        self.macroclusters = [self.disable_microclusters(ma) for ma in components]

    def fit(self, x_data: np.ndarray):
        lenx = len(x_data)
        for i, x in enumerate(x_data):
            self.update_microclusters(x)
        self.update_macroclusters()
        return self

    def _compute_sum_normalized_typicalities_with(
        self, x: np.ndarray, microclusters_ids, microclusters_by_id
    ):
        microclusters: list[MicroClusterClean] = []
        densities: list[float] = []
        for mi_id in microclusters_ids:
            mi = microclusters_by_id.get(mi_id)
            if mi is None:
                continue
            if mi.last_density is None or mi.last_normalized_typicality is None:
                continue
            microclusters.append(mi)
            densities.append(mi.last_density)

        if not microclusters:
            return None

        sum_density = float(np.sum(densities))
        if sum_density == 0:
            return None

        for mi in microclusters:
            mi.update_microcluster(x, update=False)

        total = 0.0
        for mi in microclusters:
            if (
                mi.last_normalized_typicality is not None
                and mi.last_density is not None
            ):
                total += mi.last_normalized_typicality * mi.last_density
        total /= sum_density
        return total

    def _predict_macrocluster(self, x: np.ndarray, microclusters_by_id=None):
        microclusters_by_id = (
            self.microclusters_by_id
            if microclusters_by_id is None
            else microclusters_by_id
        )
        sum_normalized_typicalities = []
        for label, macrocluster in enumerate(self.macroclusters):
            value = self._compute_sum_normalized_typicalities_with(
                x, macrocluster, microclusters_by_id
            )
            if value is not None:
                sum_normalized_typicalities.append((label, value))
        if len(sum_normalized_typicalities) == 0:
            return -1
        return max(
            sum_normalized_typicalities,
            key=lambda item: item[1],
        )[0]

    def predict(self, x_data: np.ndarray):
        microclusters_by_id_snapshot = {
            mi_id: _clone_microcluster_for_prediction(mi)
            for mi_id, mi in self.microclusters_by_id.items()
        }
        y = []
        for x in x_data:
            label = self._predict_macrocluster(
                x, microclusters_by_id=microclusters_by_id_snapshot
            )
            y.append(label)
        return y
