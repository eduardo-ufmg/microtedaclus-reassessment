"""Configurable MicroTEDAClus engine of the model improvement experiments.

With its default options the engine is a vectorized implementation of the
original MicroTEDAClus algorithm. Every other option is one design choice
studied in the Model Improvement chapter, so an experiment changes one option
(or a few) and leaves the rest of the algorithm untouched.

The microcluster population is a structure of arrays. The part of the algorithm
that depends on the arrival order (recursive update, outlier test, vitality and
pruning) runs in one compiled kernel over those arrays. The macrocluster graph
and the prediction of a batch are array operations (NumPy and SciPy).
"""

import math

import numpy as np
from numba import njit
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import cdist

EPS = 1e-12

M_MODES = {"sigmoid": 0, "reparam": 1, "const": 2}
LIFE_MODES = {"inactive": 0, "recency": 1, "recency_inactive": 2}


@njit(cache=True)
def _sensitivity(s, m_mode, m_up, m_a, m_b):
    """Sensitivity m(S) of the outlier test.

    sigmoid: m_up / (1 + exp(-m_a (S - m_b)))                      (original, with 3, 0.007, 100)
    reparam: m_up / (1 + (m_up - 1) (19 (m_up - 1))^(-(S - 3) / (m_a - 3))),
             so that m(3) = 1 and m(m_a) = 0.95 m_up
    const:   m_up
    """
    if m_mode == 0:
        return m_up / (1.0 + math.exp(-m_a * (s - m_b)))
    if m_mode == 1:
        base = max(19.0 * (m_up - 1.0), 1e-9)
        log_term = -(s - 3.0) / max(m_a - 3.0, 1e-6) * math.log(base)
        if log_term > 700.0:
            return 0.0
        if log_term < -700.0:
            return m_up
        return m_up / (1.0 + (m_up - 1.0) * math.exp(log_term))
    return m_up


@njit(cache=True)
def _evaluate(i, x, count, mass, mean, variance, new_mean, variance_limit, scale, m_mode, m_up, m_a, m_b, s_max, cap_variance, chi_dim,):  # fmt: skip
    """Tentatively absorb x into microcluster i; returns (outlier, s, var, norm_ecc)."""
    dim = len(x)
    s = count[i] + 1.0
    if s_max > 0.0 and s > s_max:
        s = s_max
    ratio = (s - 1.0) / s

    sq = 0.0
    for j in range(dim):
        new_mean[j] = ratio * mean[i, j] + x[j] / s
        delta = x[j] - new_mean[j]
        sq += delta * delta
    scaled = sq * scale
    var = ratio * variance[i] + scaled / (s - 1.0)

    if var == 0.0:
        ecc = 1.0 / s
    else:
        ecc = 1.0 / s + scaled / (s * var)
    norm_ecc = ecc / 2.0

    if mass[i] + 1.0 < 3.0:
        outlier = var > variance_limit
    else:
        m = _sensitivity(s, m_mode, m_up, m_a, m_b)
        if chi_dim > 0.0:
            # dimension-aware threshold: d^2 / sigma^2 <= 1 + m sqrt(2 / dim)
            limit = (2.0 + m * math.sqrt(2.0 / chi_dim)) / (2.0 * s)
        else:
            limit = (m * m + 1.0) / (2.0 * s)
        outlier = norm_ecc > limit
        if cap_variance and var > variance_limit:
            outlier = True
    return outlier, s, var, norm_ecc


@njit(cache=True)
def _absorb(i, x, s, var, norm_ecc, clock, count, mass, mean, variance, density, life, last_seen, new_mean, fading_factor, life_mode,):  # fmt: skip
    dim = len(x)
    if life_mode == 0:
        # vitality grows with the closeness of the sample to the previous mean
        if variance[i] > 0.0:
            sq_prev = 0.0
            for j in range(dim):
                diff = x[j] - mean[i, j]
                sq_prev += diff * diff
            radius = math.sqrt(variance[i])
            life[i] += ((radius - math.sqrt(sq_prev)) / radius) * fading_factor
        else:
            life[i] = 1.0
    else:
        life[i] = 1.0

    count[i] = s
    mass[i] += 1.0
    for j in range(dim):
        mean[i, j] = new_mean[j]
    variance[i] = var
    density[i] = 1.0 / norm_ecc
    last_seen[i] = clock


@njit(cache=True)
def _fit_kernel(X, count, mass, mean, variance, density, life, active, last_seen, label, n, clock, variance_limit, fading_factor, scale, m_mode, m_up, m_a, m_b, s_max, cap_variance, life_mode, born_active, chi_dim, single_assign,):  # fmt: skip
    """Process a batch of samples sequentially; returns the new microcluster count.

    count is the effective sample count of the recursive statistics (capped at
    s_max when s_max > 0, which makes them forget); mass is the number of
    samples absorbed.
    """
    n_samples, dim = X.shape
    new_mean = np.empty(dim)

    for k in range(n_samples):
        x = X[k]
        clock += 1
        is_new = True
        best, best_dist = -1, np.inf

        for i in range(n):
            outlier, s, var, norm_ecc = _evaluate(
                i, x, count, mass, mean, variance, new_mean,
                variance_limit, scale, m_mode, m_up, m_a, m_b, s_max, cap_variance, chi_dim,
            )  # fmt: skip
            if outlier:
                continue
            is_new = False
            if single_assign:
                sq_prev = 0.0
                for j in range(dim):
                    diff = x[j] - mean[i, j]
                    sq_prev += diff * diff
                if sq_prev < best_dist:
                    best, best_dist = i, sq_prev
            else:
                _absorb(
                    i, x, s, var, norm_ecc, clock, count, mass, mean, variance, density, life,
                    last_seen, new_mean, fading_factor, life_mode,
                )  # fmt: skip

        if single_assign and best >= 0:
            outlier, s, var, norm_ecc = _evaluate(
                best, x, count, mass, mean, variance, new_mean,
                variance_limit, scale, m_mode, m_up, m_a, m_b, s_max, cap_variance, chi_dim,
            )  # fmt: skip
            _absorb(
                best, x, s, var, norm_ecc, clock, count, mass, mean, variance, density, life,
                last_seen, new_mean, fading_factor, life_mode,
            )  # fmt: skip

        if is_new:
            count[n] = 1.0
            mass[n] = 1.0
            for j in range(dim):
                mean[n, j] = x[j]
            variance[n] = 0.0
            density[n] = 0.0
            life[n] = 1.0
            active[n] = born_active
            last_seen[n] = clock
            label[n] = -1
            n += 1

        # Vitality decay and pruning, compacting the arrays in place
        kept = 0
        for i in range(n):
            if life_mode == 0:
                decays = not active[i]
            elif life_mode == 1:
                decays = last_seen[i] != clock
            else:
                decays = (not active[i]) and last_seen[i] != clock
            if decays:
                life[i] -= fading_factor
                if life[i] < 0.0:
                    continue
            if kept != i:
                count[kept] = count[i]
                mass[kept] = mass[i]
                for j in range(dim):
                    mean[kept, j] = mean[i, j]
                variance[kept] = variance[i]
                density[kept] = density[i]
                life[kept] = life[i]
                active[kept] = active[i]
                last_seen[kept] = last_seen[i]
                label[kept] = label[i]
            kept += 1
        n = kept

    return n, clock


def _max_per_group(values, group, n_groups):
    """Row-wise maximum of the columns of each group (groups are column labels)."""
    order = np.argsort(group, kind="stable")
    starts = np.searchsorted(group[order], np.arange(n_groups))
    return np.maximum.reduceat(values[:, order], starts, axis=1)


class MicroTEDAClusEngine:
    """MicroTEDAClus with one switch per design choice under study.

    Microcluster level
        dim_penalty      distances are scaled by 2 / d (original) or not
        born_active      a new microcluster is flagged active (original) or inactive
        m_mode           sensitivity function: sigmoid (original) | reparam | const
        m_up, m_a, m_b   parameters of the sensitivity function
        s_max            > 0 caps the effective count of the statistics (forgetting)
        cap_variance     variance_limit also bounds microclusters with three or more samples
        life_mode        which microclusters decay at each sample:
                         inactive (original) | recency | recency_inactive
        chi_dim          dimension-aware outlier threshold
        single_assign    a sample updates only the nearest accepting microcluster
        relative_limit   variance_limit is a fraction of the stream's running variance
        shrink_exp       > 0 shrinks variance_limit as (n / 100)^-shrink_exp
    Macrocluster level
        density          microcluster density: last (original) | peak | volume
        activation       mean (original: density >= factor * component mean) | none
        activation_scope members that enter the component mean: all (original) | mature
        activation_factor
        link             microclusters intersect if dist <= link (sigma_i + sigma_j)
        topology         macroclusters from the active graph g1 (original) or from g0
        membership       typicality (original) | norm_typicality | local_weights |
                         density_sum | density_max | nearest
        noise_m          > 0 labels -1 the samples beyond noise_m sigma of every active microcluster
        stable_labels    macroclusters keep their label across updates
    """

    _FIELDS = ("count", "mass", "mean", "variance", "density", "life", "active", "last_seen", "label")  # fmt: skip
    _DTYPES = {"active": bool, "last_seen": np.int64, "label": np.int64}

    def __init__(
        self,
        variance_limit=0.001,
        decay=100,
        *,
        dim_penalty=True,
        born_active=True,
        m_mode="sigmoid",
        m_up=3.0,
        m_a=0.007,
        m_b=100.0,
        s_max=0.0,
        cap_variance=False,
        life_mode="inactive",
        chi_dim=False,
        single_assign=False,
        relative_limit=False,
        shrink_exp=0.0,
        density="last",
        activation="mean",
        activation_scope="all",
        activation_factor=1.0,
        link=2.0,
        topology="g1",
        membership="typicality",
        noise_m=0.0,
        stable_labels=False,
        epsilon=1e-8,
    ):
        self.variance_limit = float(variance_limit)
        self.fading_factor = 1.0 / decay
        self.total_num_samples = 0

        self.dim_penalty = dim_penalty
        self.born_active = born_active
        self.m_mode = M_MODES[m_mode]
        self.m_up, self.m_a, self.m_b = float(m_up), float(m_a), float(m_b)
        self.s_max = float(s_max)
        self.cap_variance = cap_variance
        self.life_mode = LIFE_MODES[life_mode]
        self.chi_dim = chi_dim
        self.single_assign = single_assign
        self.relative_limit = relative_limit
        self.shrink_exp = float(shrink_exp)

        self.density = density
        self.activation = activation
        self.activation_scope = activation_scope
        self.activation_factor = activation_factor
        self.link = link
        self.topology = topology
        self.membership = membership
        self.noise_m = noise_m
        self.stable_labels = stable_labels
        self.epsilon = epsilon

        self._n = 0
        self._clock = 0
        self._dim = None
        self._arrays = {}
        self._pred = None
        self._next_label = 0
        self._stream_count = 0
        self._stream_sum = None
        self._stream_sq = 0.0
        self.n_macro_ = 0

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------
    def _field_shape(self, name, capacity):
        return (capacity, self._dim) if name == "mean" else (capacity,)

    def _reserve(self, extra, dim):
        """Guarantee room for `extra` new microclusters (at most one per sample)."""
        n = self._n
        if self._dim is not None and n + extra <= len(self._arrays["mass"]):
            return
        self._dim = dim
        capacity = max(64, 2 * (n + extra))
        grown = {}
        for name in self._FIELDS:
            arr = np.zeros(
                self._field_shape(name, capacity), dtype=self._DTYPES.get(name, float)
            )
            if name in self._arrays:
                arr[:n] = self._arrays[name][:n]
            grown[name] = arr
        self._arrays = grown

    def _view(self, name):
        return self._arrays[name][: self._n]

    # ------------------------------------------------------------------
    # Microclusters
    # ------------------------------------------------------------------
    def _scale(self):
        return (2.0 / self._dim) ** 2 if self.dim_penalty else 1.0

    def _current_limit(self, X):
        """Variance limit in force for a batch (absolute, or relative to the stream's variance)."""
        limit = self.variance_limit
        if self.shrink_exp > 0:
            seen = max(self.total_num_samples + len(X), 100)
            limit *= (seen / 100.0) ** (-self.shrink_exp)
        if not self.relative_limit:
            return limit
        if self._stream_sum is None:
            self._stream_sum = np.zeros(X.shape[1])
        self._stream_count += len(X)
        self._stream_sum += X.sum(axis=0)
        self._stream_sq += float(np.einsum("ij,ij->", X, X))
        stream_mean = self._stream_sum / self._stream_count
        stream_var = self._stream_sq / self._stream_count - float(
            stream_mean @ stream_mean
        )
        return limit * max(stream_var, EPS) * self._scale()

    def _fit_batch(self, X):
        a = self._arrays
        self._n, self._clock = _fit_kernel(
            X, a["count"], a["mass"], a["mean"], a["variance"], a["density"], a["life"],
            a["active"], a["last_seen"], a["label"], self._n, self._clock,
            self._current_limit(X), self.fading_factor, self._scale(),
            self.m_mode, self.m_up, self.m_a, self.m_b,
            self.s_max, self.cap_variance, self.life_mode, self.born_active,
            float(X.shape[1]) if self.chi_dim else 0.0, self.single_assign,
        )  # fmt: skip

    def fit(self, X):
        X = np.ascontiguousarray(X, dtype=float)
        if len(X) == 0:
            return
        self._reserve(len(X), X.shape[1])
        self._fit_batch(X)
        self.total_num_samples += len(X)
        self.update_macro_clusters()

    # ------------------------------------------------------------------
    # Macroclusters
    # ------------------------------------------------------------------
    def _microcluster_density(self):
        mass, variance = self._view("mass"), self._view("variance")
        if self.density == "last":  # 1 / zeta of the last absorbed sample
            return self._view("density").copy()
        if self.density == "peak":  # 1 / zeta at the microcluster's own mean
            return np.where(mass >= 3, 2.0 * self._view("count"), 0.0)
        if self.density == "volume":
            return np.where(mass >= 3, mass / (variance + self.epsilon), 0.0)
        raise ValueError(self.density)

    def _adjacency(self, mean, std):
        adjacency = cdist(mean, mean) <= self.link * (std[:, None] + std[None, :])
        np.fill_diagonal(adjacency, False)
        return adjacency

    def update_macro_clusters(self):
        n = self._n
        self._pred = None
        self.n_macro_ = 0
        if n == 0:
            return

        mean, variance, mass = (
            self._view("mean"),
            self._view("variance"),
            self._view("mass"),
        )
        density = self._microcluster_density()
        mature = mass > 2

        adjacency = self._adjacency(mean, np.sqrt(variance))
        _, component = connected_components(csr_matrix(adjacency), directed=False)

        if self.activation == "none":
            active = mature.copy()
        else:
            if self.activation_scope == "mature":
                members = np.bincount(component, weights=mature.astype(float))
                total = np.bincount(component, weights=np.where(mature, density, 0.0))
            else:
                members = np.bincount(component).astype(float)
                total = np.bincount(component, weights=density)
            mean_density = total / np.maximum(members, 1.0)
            active = mature & (
                density >= self.activation_factor * mean_density[component]
            )
        self._arrays["active"][:n] = active

        index = np.flatnonzero(active)
        if len(index) == 0:
            return

        if self.topology == "g1":
            n_macro, macro = connected_components(
                csr_matrix(adjacency[np.ix_(index, index)]), directed=False
            )
        else:
            _, macro = np.unique(component[index], return_inverse=True)
            n_macro = int(macro.max()) + 1

        labels = np.arange(n_macro)
        if self.stable_labels:
            labels = self._persistent_labels(index, macro, n_macro, mass)

        indicator = np.zeros((len(index), n_macro))
        indicator[np.arange(len(index)), macro] = 1.0
        self.n_macro_ = n_macro
        self._pred = {
            "index": index,
            "mean": mean[index].copy(),
            "count": self._view("count")[index].copy(),
            "mass": mass[index].copy(),
            "variance": variance[index].copy(),
            "density": density[index].copy(),
            "macro": macro,
            "labels": labels,
            "indicator": indicator,
        }

    def _persistent_labels(self, index, macro, n_macro, mass):
        """Each macrocluster keeps the label that owns most of its mass; the others get a new one."""
        label = self._arrays["label"]
        previous = label[index]
        labels = np.full(n_macro, -1, dtype=np.int64)
        known = previous >= 0
        if known.any():
            shared = {}
            for c, lab, w in zip(macro[known], previous[known], mass[index][known]):
                shared[(c, lab)] = shared.get((c, lab), 0.0) + w
            taken = set()
            for (c, lab), _ in sorted(shared.items(), key=lambda item: -item[1]):
                if labels[c] < 0 and lab not in taken:
                    labels[c] = lab
                    taken.add(lab)
        for c in range(n_macro):
            if labels[c] < 0:
                labels[c] = self._next_label
                self._next_label += 1
        label[index] = labels[macro]
        return labels

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def _standardized_sq_distance(self, X, p):
        """d^2 / sigma^2 to each active microcluster (inf for a zero variance unless d = 0)."""
        scaled = cdist(X, p["mean"], "sqeuclidean") * self._scale()
        positive = p["variance"] > 0
        std_sq = np.where(scaled > 0, np.inf, 0.0)
        std_sq[:, positive] = scaled[:, positive] / p["variance"][positive]
        return std_sq

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        p = self._pred
        if p is None:
            self.labels_ = np.full(len(X), -1, dtype=int)
            return self.labels_

        s, indicator, macro = p["count"], p["indicator"], p["macro"]
        std_sq = self._standardized_sq_distance(X, p)
        ecc = 1.0 / s + np.where(p["variance"] > 0, std_sq, 0.0) / s

        mode = self.membership
        if mode in ("typicality", "norm_typicality"):
            weights = (
                indicator
                * (p["density"] / (indicator.T @ p["density"])[macro])[:, None]
            )
            typicality = (
                1.0 - ecc / 2.0 if mode == "typicality" else (1.0 - ecc) / (s - 2.0)
            )
            scores = typicality @ weights
        elif mode == "local_weights":
            local_density = 2.0 * s / (1.0 + std_sq)  # 1 / zeta(x)
            weighted = (local_density * ((1.0 - ecc) / (s - 2.0))) @ indicator
            scores = weighted / np.maximum(local_density @ indicator, EPS)
        elif mode == "density_sum":
            scores = (2.0 * p["mass"] / (1.0 + std_sq)) @ indicator
        elif mode == "density_max":
            scores = _max_per_group(
                2.0 * p["mass"] / (1.0 + std_sq), macro, indicator.shape[1]
            )
        elif mode == "nearest":
            scores = _max_per_group(1.0 / (1.0 + std_sq), macro, indicator.shape[1])
        else:
            raise ValueError(mode)

        labels = p["labels"][np.argmax(scores, axis=1)]
        if self.noise_m > 0:
            labels = np.where(std_sq.min(axis=1) > self.noise_m**2, -1, labels)
        self.labels_ = labels
        return labels

    def microcluster_count(self):
        return self._n
