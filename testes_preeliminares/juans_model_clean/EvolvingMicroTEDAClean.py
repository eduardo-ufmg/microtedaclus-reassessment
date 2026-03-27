from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx
import numpy as np
from numba import njit
from scipy.spatial.distance import cdist


@njit(cache=True)
def jit_evaluate(
    x: np.ndarray,
    mean: np.ndarray,
    variance: float,
    num_samples: int,
    resolution: float,
    m: float = 3.0,
) -> tuple[float, float, bool]:
    diff = x - mean
    dist_sq = np.dot(diff, diff)
    res_sq = resolution**2

    # Cold-start handling for statistically insignificant clusters
    if num_samples < 3:
        is_outlier = dist_sq > res_sq
        eccentricity = dist_sq / res_sq
        typicality = 1.0 / (1.0 + eccentricity)
        return typicality, eccentricity, is_outlier

    # Standard TEDA with clamped variance floor
    eff_var = max(variance, res_sq)
    n_plus_1 = num_samples + 1

    eccentricity = (1.0 / n_plus_1) + (dist_sq / (n_plus_1 * eff_var))
    typicality = 1.0 / (1.0 + n_plus_1 * eccentricity)

    threshold = (m**2 + 1.0) / n_plus_1
    is_outlier = eccentricity > threshold

    return typicality, eccentricity, is_outlier


@njit(cache=True)
def jit_welford_update(
    mean: np.ndarray, s: float, x: np.ndarray, num_samples: int
) -> tuple[np.ndarray, float, float]:
    """Computes exact online scalar variance using Welford's algorithm."""
    n_new = num_samples + 1
    diff = x - mean
    mean_new = mean + diff / n_new
    s_new = s + np.dot(diff, x - mean_new)
    var_new = s_new / n_new
    return mean_new, s_new, var_new


@dataclass
class MicroClusterClean:
    id: int
    mean: np.ndarray
    resolution: float
    num_samples: int = 1
    s: float = 0.0  # Welford's sum of squared differences
    variance: float = 0.0
    inactive_steps: int = 0

    @property
    def density(self) -> float:
        return self.num_samples / max(self.variance, self.resolution**2)

    def evaluate(self, x: np.ndarray) -> tuple[float, float, bool]:
        return jit_evaluate(
            x, self.mean, self.variance, self.num_samples, self.resolution
        )

    def update(self, x: np.ndarray) -> None:
        self.mean, self.s, self.variance = jit_welford_update(
            self.mean, self.s, x, self.num_samples
        )
        self.num_samples += 1
        self.inactive_steps = 0


@dataclass
class EvolvingMicroTEDAClean:
    resolution: float
    memory: int
    microclusters_by_id: dict[int, MicroClusterClean] = field(default_factory=dict)
    macroclusters: list[list[int]] = field(default_factory=list)
    graph: nx.Graph = field(default_factory=nx.Graph)
    counter_microclusters: int = 0

    @property
    def microclusters(self) -> list[MicroClusterClean]:
        return list(self.microclusters_by_id.values())

    def create_microcluster(self, x: np.ndarray) -> None:
        new_microcluster = MicroClusterClean(
            id=self.counter_microclusters, mean=x.copy(), resolution=self.resolution
        )
        self.microclusters_by_id[new_microcluster.id] = new_microcluster
        self.graph.add_node(new_microcluster.id)
        self.counter_microclusters += 1

    def purge_forgotten_microclusters(self) -> None:
        dead_ids = [
            mi_id
            for mi_id, mi in self.microclusters_by_id.items()
            if mi.inactive_steps >= self.memory
        ]
        for mi_id in dead_ids:
            if mi_id in self.graph:
                self.graph.remove_node(mi_id)
            del self.microclusters_by_id[mi_id]

    def update_microclusters(self, x: np.ndarray) -> None:
        if not self.microclusters_by_id:
            self.create_microcluster(x)
            return

        best_mi: MicroClusterClean | None = None
        max_typicality = -1.0
        best_is_outlier = True

        for mi in self.microclusters_by_id.values():
            typicality, _, is_outlier = mi.evaluate(x)
            if typicality > max_typicality:
                max_typicality = typicality
                best_mi = mi
                best_is_outlier = is_outlier

        if best_mi is not None and not best_is_outlier:
            best_mi.update(x)
            for mi in self.microclusters_by_id.values():
                if mi.id != best_mi.id:
                    mi.inactive_steps += 1
        else:
            self.create_microcluster(x)
            for mi in self.microclusters_by_id.values():
                if mi.id != self.counter_microclusters - 1:
                    mi.inactive_steps += 1

        self.purge_forgotten_microclusters()

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
                if mi.id in self.graph:
                    self.graph.remove_node(mi.id)
                    self.graph.add_node(mi.id)

        return valid_ids

    def update_macroclusters(self) -> None:
        mcs = self.microclusters
        count = len(mcs)
        if count == 0:
            self.macroclusters = []
            return

        means = np.array([mi.mean for mi in mcs])
        variances = np.array([max(mi.variance, self.resolution**2) for mi in mcs])
        ids = [mi.id for mi in mcs]

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

    def fit(self, x_data: np.ndarray) -> EvolvingMicroTEDAClean:
        for x in x_data:
            self.update_microclusters(x)
        self.update_macroclusters()
        return self

    def predict(self, x_data: np.ndarray) -> list[int]:
        predictions = []
        for x in x_data:
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

            predictions.append(best_label)
        return predictions
