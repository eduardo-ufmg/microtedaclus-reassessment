from dataclasses import dataclass
import math
import networkx as nx
import numpy as np
import optuna
from scipy.spatial.distance import cdist


@dataclass(slots=True)
class MicroCluster:
    id: int
    num_samples: int
    mean: np.ndarray
    variance: float
    density: float
    life: float
    changed: bool = True


class MicroTEDAClusImproved_const_up:
    """MicroTEDAClus with the reparameterized m(k) function fixing up = 3 constant,
    exposing only k_95 as a tunable parameter:
        m(k) = 3 / (1 + 2 * (38)^(- (k - 3) / (k_95 - 3)))
    where k_95 > 3.
    """

    def __init__(
        self,
        variance_limit: float = 0.001,
        decay: float = 100.0,
        k_95: float = 500.0,
        epsilon: float = 1e-8,
    ):
        self.variance_limit = float(variance_limit)
        self.decay = float(decay)
        self.fading_factor = 1.0 / self.decay
        self.up = 3.0
        self.k_95 = float(k_95)
        self.epsilon = float(epsilon)
        self.total_num_samples = 0

        self.micro_clusters: list[MicroCluster] = []
        self.micro_clusters_by_id: dict[int, MicroCluster] = {}
        self.macro_clusters: list[set[int]] = []
        self.graph = nx.Graph()
        self.changed_micro_cluster_ids: set[int] = set()
        self._next_id = 1

    def create_new_micro_cluster(self, x: np.ndarray) -> None:
        mc_id = self._next_id
        self._next_id += 1

        mean = np.array(x, dtype=float, copy=True)
        variance = 0.0
        num_samples = 1
        density = 1.0 / self.epsilon
        life = 1.0

        mc = MicroCluster(
            id=mc_id,
            num_samples=num_samples,
            mean=mean,
            variance=variance,
            density=density,
            life=life,
            changed=True,
        )
        self.micro_clusters.append(mc)
        self.micro_clusters_by_id[mc_id] = mc
        self.changed_micro_cluster_ids.add(mc_id)
        self.graph.add_node(mc_id)

    def is_outlier(self, s_ik: int, var_ik: float, norm_ecc: float) -> bool:
        if s_ik < 3:
            return var_ik > self.variance_limit
        else:
            denom_k = max(self.k_95 - 3.0, 1e-6)
            exponent = -(s_ik - 3.0) / denom_k
            log_term = exponent * math.log(38.0)
            if log_term > 700:
                mik_sik = 0.0
            elif log_term < -700:
                mik_sik = 3.0
            else:
                mik_sik = 3.0 / (1.0 + 2.0 * math.exp(log_term))

            outlier_limit = ((mik_sik**2) + 1.0) / (2.0 * s_ik)
            return norm_ecc > outlier_limit

    def update_micro_clusters(self, xk: np.ndarray) -> None:
        if self.total_num_samples == 0:
            self.create_new_micro_cluster(xk)
            return

        new_micro_cluster = True

        for mi in self.micro_clusters:
            mi.changed = False
            s_old = mi.num_samples
            mu_old = mi.mean
            var_old = mi.variance

            s_new = s_old + 1
            mean_new = ((s_new - 1) / s_new) * mu_old + (xk / s_new)

            delta = xk - mean_new
            norm_delta_sq = np.dot(delta, delta)
            var_new = ((s_new - 1) / s_new) * var_old + (norm_delta_sq / (s_new - 1))

            if var_new == 0.0 and s_new > 1:
                ecc = 1.0 / s_new
            else:
                a = mean_new - xk
                ecc = (1.0 / s_new) + (np.dot(a, a) / (s_new * var_new))
            norm_ecc = ecc * 0.5

            if not self.is_outlier(s_new, var_new, norm_ecc):
                mi.num_samples = s_new
                mi.mean = mean_new
                mi.variance = var_new
                mi.density = s_new / (var_new + self.epsilon)
                mi.life = 1.0
                mi.changed = True
                self.changed_micro_cluster_ids.add(mi.id)
                new_micro_cluster = False
            else:
                mi.life -= self.fading_factor / s_old

        if new_micro_cluster:
            self.create_new_micro_cluster(xk)

    def prune_micro_clusters(self) -> None:
        surviving = []
        for mc in self.micro_clusters:
            if mc.life <= 0:
                del self.micro_clusters_by_id[mc.id]
                self.changed_micro_cluster_ids.discard(mc.id)
                if self.graph.has_node(mc.id):
                    self.graph.remove_node(mc.id)
            else:
                surviving.append(mc)
        self.micro_clusters = surviving

    def define_macro_clusters(self) -> None:
        if not self.changed_micro_cluster_ids or len(self.micro_clusters) < 2:
            self.macro_clusters = list(nx.connected_components(self.graph))
            self.changed_micro_cluster_ids.clear()
            return

        changed_mcs = [
            self.micro_clusters_by_id[cid]
            for cid in self.changed_micro_cluster_ids
            if cid in self.micro_clusters_by_id
        ]
        if not changed_mcs:
            self.macro_clusters = list(nx.connected_components(self.graph))
            self.changed_micro_cluster_ids.clear()
            return

        c_ids = np.array([m.id for m in changed_mcs])
        c_means = np.array([m.mean for m in changed_mcs])
        c_vars = np.array([m.variance for m in changed_mcs])

        all_ids = np.array([m.id for m in self.micro_clusters])
        all_means = np.array([m.mean for m in self.micro_clusters])
        all_vars = np.array([m.variance for m in self.micro_clusters])

        dist_matrix = cdist(c_means, all_means)
        dev_matrix = 2.0 * (np.sqrt(c_vars)[:, None] + np.sqrt(all_vars)[None, :])
        intersect_matrix = dist_matrix <= dev_matrix

        to_add = []
        to_remove = []
        C, M = intersect_matrix.shape
        for i in range(C):
            u = int(c_ids[i])
            for j in range(M):
                v = int(all_ids[j])
                if u == v:
                    continue
                if intersect_matrix[i, j]:
                    to_add.append((u, v))
                elif self.graph.has_edge(u, v):
                    to_remove.append((u, v))

        self.graph.add_edges_from(to_add)
        self.graph.remove_edges_from(to_remove)
        self.macro_clusters = list(nx.connected_components(self.graph))
        self.changed_micro_cluster_ids.clear()

    def fit(
        self,
        X: np.ndarray,
        update_macro_clusters: bool = True,
        prune_micro_clusters: bool = True,
    ) -> None:
        for xk in X:
            self.update_micro_clusters(xk)
            if prune_micro_clusters:
                self.prune_micro_clusters()
            self.total_num_samples += 1

        if update_macro_clusters:
            self.define_macro_clusters()

    def predict(self, X: np.ndarray) -> np.ndarray:
        N = len(X)
        K = len(self.macro_clusters)
        if K == 0 or len(self.micro_clusters) == 0:
            self.labels_ = np.full(N, -1, dtype=int)
            return self.labels_

        macro_info = []
        for mg in self.macro_clusters:
            mcs = [
                self.micro_clusters_by_id[mid]
                for mid in mg
                if mid in self.micro_clusters_by_id
            ]
            if not mcs:
                macro_info.append(None)
                continue

            total_density = sum(m.density for m in mcs)
            mean_density = total_density / len(mcs)
            active_mcs = [
                m for m in mcs if m.num_samples > 2 and m.density >= mean_density
            ]

            if not active_mcs:
                macro_info.append(None)
                continue

            act_total_density = sum(m.density for m in active_mcs)
            if act_total_density == 0:
                macro_info.append(None)
                continue

            means = np.array([m.mean for m in active_mcs])
            vars_ = np.array([m.variance for m in active_mcs])
            nums_ = np.array([m.num_samples for m in active_mcs])
            dens_ = np.array([m.density for m in active_mcs])
            weights = dens_ / act_total_density

            macro_info.append(
                {
                    "means": means,
                    "vars": vars_,
                    "nums": nums_,
                    "weights": weights,
                }
            )

        all_memberships = np.zeros((N, K), dtype=float)
        for k_idx, info in enumerate(macro_info):
            if info is None:
                continue
            means = info["means"]
            vars_ = info["vars"]
            nums_ = info["nums"]
            weights = info["weights"]

            diff = X[:, None, :] - means[None, :, :]
            sq_dist = np.sum(diff**2, axis=2)

            with np.errstate(divide="ignore", invalid="ignore"):
                term2 = sq_dist / (nums_[None, :] * vars_[None, :])
                term2 = np.where(vars_[None, :] == 0, 0.0, term2)
                ecc = (1.0 / nums_[None, :]) + term2
                norm_typ = (1.0 - ecc) / (nums_[None, :] - 2.0)

            all_memberships[:, k_idx] = np.dot(norm_typ, weights)

        self.labels_ = np.argmax(all_memberships, axis=1)
        return self.labels_

    def microcluster_count(self) -> int:
        return len(self.micro_clusters)


class MicroTEDAClusImproved_const_upParameterSearchSpace:
    def sample_parameters(self, trial: optuna.Trial) -> dict:
        return {
            "variance_limit": trial.suggest_float(
                "variance_limit", 0.0001, 100.0, log=True
            ),
            "decay": trial.suggest_int("decay", 10, 10000, log=True),
            "k_95": trial.suggest_float("k_95", 5.0, 5000.0, log=True),
        }
