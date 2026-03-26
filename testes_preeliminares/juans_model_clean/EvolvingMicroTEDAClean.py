from __future__ import annotations

import math
from dataclasses import dataclass, field

import networkx as nx
import numpy as np
from numba import njit
from scipy.spatial.distance import cdist

EPSILON = 1e-12


@njit(cache=True)
def jit_compute_new_mean(
    mean: np.ndarray, x: np.ndarray, num_samples: int
) -> np.ndarray:
    n_plus_1 = num_samples + 1
    return (num_samples / n_plus_1) * mean + x / n_plus_1


@njit(cache=True)
def jit_compute_new_variance(
    variance: float, x: np.ndarray, new_mean: np.ndarray, num_samples: int
) -> float:
    if num_samples <= 0:
        return 0.0
    diff = x - new_mean
    sq_norm = np.dot(diff, diff)
    n_plus_1 = num_samples + 1
    # Standard recursive scalar variance
    return (num_samples / n_plus_1) * variance + (1.0 / num_samples) * sq_norm


@njit(cache=True)
def jit_compute_eccentricity(
    x: np.ndarray, mean: np.ndarray, variance: float, num_samples: int
) -> float:
    n_plus_1 = num_samples + 1
    if variance <= EPSILON:
        diff = x - mean
        if np.dot(diff, diff) <= EPSILON:
            return 1.0 / n_plus_1
        return 1e6  # Arbitrary large penalty for distant points when variance is zero

    diff = x - mean
    sq_norm = np.dot(diff, diff)
    return (1.0 / n_plus_1) + (sq_norm / (n_plus_1 * variance))


@njit(cache=True)
def jit_compute_typicality(eccentricity: float, num_samples: int) -> float:
    # Strictly bounded typicality in [0, 1]
    return 1.0 / (1.0 + (num_samples + 1) * eccentricity)


@njit(cache=True)
def jit_is_outlier(
    eccentricity: float,
    num_samples: int,
    variance: float,
    variance_limit: float,
    m: float = 3.0,
) -> bool:
    if num_samples < 3:
        return variance > variance_limit
    # Chebyshev-derived bound for TEDA
    threshold = (m**2 + 1.0) / (num_samples + 1)
    return eccentricity > threshold


@njit(cache=True)
def jit_has_intersection(
    mean1: np.ndarray, mean2: np.ndarray, var1: float, var2: float
) -> bool:
    diff = mean1 - mean2
    dist = np.sqrt(np.dot(diff, diff))
    threshold = 2.0 * (np.sqrt(var1) + np.sqrt(var2))
    return dist < threshold


@dataclass
class MicroCluster:
    id: int
    mean: np.ndarray
    variance_limit: float = 0.01
    num_samples: int = 1
    variance: float = 0.0
    life: float = 1.0
    decay: float = 1000.0

    @property
    def density(self) -> float:
        return self.num_samples / (self.variance + EPSILON)

    def evaluate(self, x: np.ndarray) -> tuple[float, float, bool]:
        """Stateless evaluation of a point against this cluster."""
        eccentricity = jit_compute_eccentricity(
            x, self.mean, self.variance, self.num_samples
        )
        typicality = jit_compute_typicality(eccentricity, self.num_samples)

        # Lookahead variance to check variance limit constraints
        new_mean = jit_compute_new_mean(self.mean, x, self.num_samples)
        new_variance = jit_compute_new_variance(
            self.variance, x, new_mean, self.num_samples
        )

        is_outlier = jit_is_outlier(
            eccentricity, self.num_samples, new_variance, self.variance_limit
        )
        return typicality, eccentricity, is_outlier

    def update(self, x: np.ndarray) -> None:
        """Mutates state by absorbing the given point."""
        new_mean = jit_compute_new_mean(self.mean, x, self.num_samples)
        new_variance = jit_compute_new_variance(
            self.variance, x, new_mean, self.num_samples
        )
        self.mean = new_mean
        self.variance = new_variance
        self.num_samples += 1
        self.life = 1.0  # Reset life on successful absorption

    def decline_life(self) -> None:
        if self.decay > 0:
            self.life -= 1.0 / self.decay


@dataclass
class MTEDAC:
    variance_limit: float = 0.01
    decay: float = 1000.0
    microclusters_by_id: dict[int, MicroCluster] = field(default_factory=dict)
    macroclusters: list[list[int]] = field(default_factory=list)
    graph: nx.Graph = field(default_factory=nx.Graph)
    counter_microclusters: int = 0

    @property
    def microclusters(self) -> list[MicroCluster]:
        return list(self.microclusters_by_id.values())

    def create_microcluster(self, x: np.ndarray) -> None:
        new_microcluster = MicroCluster(
            id=self.counter_microclusters,
            mean=x.copy(),
            variance_limit=self.variance_limit,
            decay=self.decay,
        )
        self.microclusters_by_id[new_microcluster.id] = new_microcluster
        self.graph.add_node(new_microcluster.id)
        self.counter_microclusters += 1

    def remove_dead_microclusters(self) -> None:
        dead_ids = [
            mi_id for mi_id, mi in self.microclusters_by_id.items() if mi.life <= 0
        ]
        for mi_id in dead_ids:
            if mi_id in self.graph:
                self.graph.remove_node(mi_id)
            del self.microclusters_by_id[mi_id]

    def update_microclusters(self, x: np.ndarray) -> None:
        if not self.microclusters_by_id:
            self.create_microcluster(x)
            return

        best_mi: MicroCluster | None = None
        max_typicality = -1.0
        best_is_outlier = True

        # Stateless evaluation for Winner-Takes-All assignment
        for mi in self.microclusters_by_id.values():
            typicality, _, is_outlier = mi.evaluate(x)
            if typicality > max_typicality:
                max_typicality = typicality
                best_mi = mi
                best_is_outlier = is_outlier

        if best_mi is not None and not best_is_outlier:
            best_mi.update(x)
            # Decay others
            for mi in self.microclusters_by_id.values():
                if mi.id != best_mi.id:
                    mi.decline_life()
        else:
            self.create_microcluster(x)
            for mi in self.microclusters_by_id.values():
                mi.decline_life()

        self.remove_dead_microclusters()

    def filter_low_density_components(self, component: set[int]) -> list[int]:
        microclusters = [
            self.microclusters_by_id[mi_id]
            for mi_id in component
            if mi_id in self.microclusters_by_id
        ]
        if not microclusters:
            return []

        mean_density = float(np.mean([mi.density for mi in microclusters]))

        valid_ids = []
        for mi in microclusters:
            if mi.density >= mean_density:
                valid_ids.append(mi.id)
            else:
                # Disconnect low density nodes from the topology
                if mi.id in self.graph:
                    self.graph.remove_node(mi.id)
                    self.graph.add_node(mi.id)  # Re-add as isolated node

        return valid_ids

    def update_macroclusters(self) -> None:
        mcs = self.microclusters
        count = len(mcs)
        if count == 0:
            self.macroclusters = []
            return

        means = np.array([mi.mean for mi in mcs])
        variances = np.array([mi.variance for mi in mcs])
        ids = [mi.id for mi in mcs]

        # Topology edge updates
        if count > 1:
            distances = cdist(means, means, metric="euclidean")
            for i in range(count):
                for j in range(i + 1, count):
                    threshold = 2 * (np.sqrt(variances[i]) + np.sqrt(variances[j]))
                    edge = (ids[i], ids[j])
                    if distances[i, j] < threshold:
                        self.graph.add_edge(*edge)
                    elif self.graph.has_edge(*edge):
                        self.graph.remove_edge(*edge)

        components = list(nx.connected_components(self.graph))
        self.macroclusters = [
            self.filter_low_density_components(comp) for comp in components if comp
        ]
        self.macroclusters = [ma for ma in self.macroclusters if ma]

    def fit(self, x_data: np.ndarray) -> MTEDAC:
        for x in x_data:
            self.update_microclusters(x)
        self.update_macroclusters()
        return self

    def _predict_macrocluster(self, x: np.ndarray) -> int:
        best_label = -1
        max_score = -1.0

        for label, macrocluster_ids in enumerate(self.macroclusters):
            score = 0.0
            density_sum = 0.0

            for mi_id in macrocluster_ids:
                if mi_id not in self.microclusters_by_id:
                    continue
                mi = self.microclusters_by_id[mi_id]
                typicality, _, _ = mi.evaluate(x)

                score += typicality * mi.density
                density_sum += mi.density

            if density_sum > 0:
                normalized_score = score / density_sum
                if normalized_score > max_score:
                    max_score = normalized_score
                    best_label = label

        return best_label

    def predict(self, x_data: np.ndarray) -> list[int]:
        return [self._predict_macrocluster(x) for x in x_data]
