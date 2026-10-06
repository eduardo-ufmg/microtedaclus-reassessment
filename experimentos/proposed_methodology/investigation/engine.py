"""Configurable research engine for the MicroTEDAClus investigation.

Every option defaults to the behaviour of the reference implementation, so
`TedaClus()` is equivalent to MicroTEDAClusVectorized. Each correction or
improvement under study is a switch, which lets the harness measure one change
at a time. The sequential microcluster logic runs in a compiled kernel; the
macrocluster logic and the prediction are vectorized with NumPy/SciPy.
"""

import math

import numpy as np
from numba import njit
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import cdist

EPS = 1e-12


@njit(cache=True)
def _m_value(s, m_mode, m_up, m_a, m_b):
    """Sensitivity m(S) of the outlier test.

    0: reference sigmoid  m_up / (1 + exp(-m_a (S - m_b)))
    1: reparameterized    m_up / (1 + (m_up - 1) (19 (m_up - 1))^(-(S - 3) / (m_a - 3))), m(3) = 1, m(m_a) = 0.95 m_up
    2: constant           m_up
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
def _evaluate(
    i,
    x,
    num_samples,
    mass,
    mean,
    variance,
    new_mean,
    variance_limit,
    var_scale,
    m_mode,
    m_up,
    m_a,
    m_b,
    s_max,
    cap_variance,
    chi_dim,
):
    """Tentatively absorb x into microcluster i; returns (outlier, s, var, norm_ecc)."""
    dim = len(x)
    s = num_samples[i] + 1.0
    if s_max > 0.0 and s > s_max:
        s = s_max
    ratio = (s - 1.0) / s

    sq = 0.0
    for j in range(dim):
        new_mean[j] = ratio * mean[i, j] + x[j] / s
        delta = x[j] - new_mean[j]
        sq += delta * delta
    if var_scale < 0.0:
        scaled = (math.sqrt(sq) * (2.0 / dim)) ** 2  # reference arithmetic
    else:
        scaled = sq * var_scale
    var = ratio * variance[i] + scaled / (s - 1.0)

    if var == 0.0:
        ecc = 1.0 / s
    else:
        ecc = 1.0 / s + scaled / (s * var)
    norm_ecc = ecc / 2.0

    if mass[i] + 1.0 < 3.0:
        outlier = var > variance_limit
    else:
        m = _m_value(s, m_mode, m_up, m_a, m_b)
        if chi_dim > 0.0:
            # dimension-aware threshold: d^2/sigma^2 <= 1 + m sqrt(2/dim)
            limit = (2.0 + m * math.sqrt(2.0 / chi_dim)) / (2.0 * s)
        else:
            limit = (m * m + 1.0) / (2.0 * s)
        outlier = norm_ecc > limit
        if cap_variance and var > variance_limit:
            outlier = True
    return outlier, s, var, norm_ecc


@njit(cache=True)
def _absorb(
    i,
    x,
    s,
    var,
    norm_ecc,
    clock,
    num_samples,
    mass,
    mean,
    variance,
    density,
    life,
    last_seen,
    new_mean,
    fading_factor,
    life_mode,
):
    dim = len(x)
    if life_mode == 0:
        # reference: distance-ratio increment relative to the previous state
        if variance[i] > 0.0:
            sq_prev = 0.0
            for j in range(dim):
                diff = x[j] - mean[i, j]
                sq_prev += diff * diff
            rt = math.sqrt(variance[i])
            life[i] += ((rt - math.sqrt(sq_prev)) / rt) * fading_factor
        else:
            life[i] = 1.0
    else:
        life[i] = 1.0

    num_samples[i] = s
    mass[i] += 1.0
    for j in range(dim):
        mean[i, j] = new_mean[j]
    variance[i] = var
    density[i] = 1.0 / norm_ecc
    last_seen[i] = clock


@njit(cache=True)
def _fit_kernel(
    X,
    num_samples,
    mass,
    mean,
    variance,
    density,
    life,
    active,
    last_seen,
    label,
    n,
    clock,
    variance_limit,
    fading_factor,
    var_scale,
    m_mode,
    m_up,
    m_a,
    m_b,
    s_max,
    cap_variance,
    life_mode,
    prune_skip,
    born_active,
    chi_dim,
    single_assign,
):
    """Process a batch of samples sequentially; returns the new microcluster count.

    num_samples is the effective count used by the recursive statistics (capped
    at s_max when s_max > 0, which turns them into exponentially forgetting
    estimates); mass is the uncapped number of absorbed samples.
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
                i,
                x,
                num_samples,
                mass,
                mean,
                variance,
                new_mean,
                variance_limit,
                var_scale,
                m_mode,
                m_up,
                m_a,
                m_b,
                s_max,
                cap_variance,
                chi_dim,
            )
            if not outlier:
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
                        i,
                        x,
                        s,
                        var,
                        norm_ecc,
                        clock,
                        num_samples,
                        mass,
                        mean,
                        variance,
                        density,
                        life,
                        last_seen,
                        new_mean,
                        fading_factor,
                        life_mode,
                    )

        if single_assign and best >= 0:
            outlier, s, var, norm_ecc = _evaluate(
                best,
                x,
                num_samples,
                mass,
                mean,
                variance,
                new_mean,
                variance_limit,
                var_scale,
                m_mode,
                m_up,
                m_a,
                m_b,
                s_max,
                cap_variance,
                chi_dim,
            )
            _absorb(
                best,
                x,
                s,
                var,
                norm_ecc,
                clock,
                num_samples,
                mass,
                mean,
                variance,
                density,
                life,
                last_seen,
                new_mean,
                fading_factor,
                life_mode,
            )

        if is_new:
            num_samples[n] = 1.0
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

        # ---- pruning -------------------------------------------------
        m_kept = 0
        skip = False
        for i in range(n):
            keep = True
            if skip:
                skip = False
            else:
                if life_mode == 0:
                    # reference: inactive microclusters lose rho per sample
                    if not active[i]:
                        life[i] -= fading_factor
                        if life[i] < 0.0:
                            keep = False
                elif life_mode == 1:
                    # recency: every microcluster not updated by this sample loses rho
                    if last_seen[i] != clock:
                        life[i] -= fading_factor
                        if life[i] < 0.0:
                            keep = False
                elif life_mode == 2:
                    # recency for inactive only; active ones are protected
                    if (not active[i]) and last_seen[i] != clock:
                        life[i] -= fading_factor
                        if life[i] < 0.0:
                            keep = False
                if (not keep) and prune_skip:
                    skip = True
            if keep:
                if m_kept != i:
                    num_samples[m_kept] = num_samples[i]
                    mass[m_kept] = mass[i]
                    for j in range(dim):
                        mean[m_kept, j] = mean[i, j]
                    variance[m_kept] = variance[i]
                    density[m_kept] = density[i]
                    life[m_kept] = life[i]
                    active[m_kept] = active[i]
                    last_seen[m_kept] = last_seen[i]
                    label[m_kept] = label[i]
                m_kept += 1
        n = m_kept

    return n, clock


class TedaClus:
    def __init__(
        self,
        variance_limit=0.001,
        decay=100,
        *,
        # --- microcluster level ---
        dim_penalty=True,  # reference scales distances by 2/d
        prune_skip=True,  # reference skips the element after a removed one
        born_active=True,  # reference creates microclusters flagged active
        m_mode="sigmoid",  # sigmoid | reparam | const
        m_up=3.0,
        m_a=0.007,
        m_b=100.0,
        s_max=0.0,  # > 0: cap of the effective count (forgetting)
        cap_variance=False,  # apply variance_limit to mature microclusters too
        life_mode="reference",  # reference | recency | recency_inactive
        chi_dim=False,  # dimension-aware outlier threshold
        single_assign=False,  # a sample updates only the nearest accepting microcluster
        relative_limit=False,  # variance_limit is a fraction of the running global variance of the stream
        shrink_exp=0.0,  # > 0: the limit shrinks as (n / 100)^-shrink_exp with the samples seen (coarse to fine)
        stable_labels=False,  # macroclusters keep their identity across updates (largest shared mass wins)
        macro_period=0,  # > 0: refresh macroclusters every N samples instead of per fit call
        # --- macrocluster level ---
        density="last",  # last | peak | snapshot
        activation="mean",  # mean | none
        activation_scope="all",  # all | mature : members that enter the component mean
        activation_factor=1.0,
        link=2.0,  # intersection: dist <= link (sigma_i + sigma_j)
        topology="g1",  # g1 | g0
        membership="reference",  # reference | norm_typ | paper | density_sum | density_max | nearest
        noise_m=0.0,  # > 0: label -1 when x is beyond noise_m sigma of every active microcluster
        epsilon=1e-8,
    ):
        self.variance_limit = float(variance_limit)
        self.fading_factor = 1.0 / decay
        self.total_num_samples = 0

        self.dim_penalty = dim_penalty
        self.prune_skip = prune_skip
        self.born_active = born_active
        self.m_mode = {"sigmoid": 0, "reparam": 1, "const": 2}[m_mode]
        self.m_up, self.m_a, self.m_b = float(m_up), float(m_a), float(m_b)
        self.s_max = float(s_max)
        self.cap_variance = cap_variance
        self.life_mode = {"reference": 0, "recency": 1, "recency_inactive": 2}[
            life_mode
        ]
        self.chi_dim = chi_dim
        self.single_assign = single_assign
        self.relative_limit = relative_limit
        self.shrink_exp = float(shrink_exp)
        self.stable_labels = stable_labels
        self._next_label = 0
        self._g_count = 0
        self._g_sum = None
        self._g_sq = 0.0
        self.macro_period = int(macro_period)

        self.density = density
        self.activation = activation
        self.activation_scope = activation_scope
        self.activation_factor = activation_factor
        self.link = link
        self.topology = topology
        self.membership = membership
        self.noise_m = noise_m
        self.epsilon = epsilon

        self._n = 0
        self._clock = 0
        self._dim = None
        self._arrays = {}
        self._pred = None
        self.n_macro_ = 0

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------
    _FIELDS = (
        "num_samples",
        "mass",
        "mean",
        "variance",
        "density",
        "life",
        "active",
        "last_seen",
        "label",
    )

    def _reserve(self, extra, dim):
        n = self._n
        if self._dim is not None and n + extra <= len(self._arrays["mass"]):
            return
        self._dim = dim
        capacity = max(64, 2 * (n + extra))
        new = {}
        for name in self._FIELDS:
            shape = (capacity, dim) if name == "mean" else capacity
            dtype = (
                bool
                if name == "active"
                else (np.int64 if name in ("last_seen", "label") else float)
            )
            arr = np.zeros(shape, dtype=dtype)
            if name in self._arrays:
                arr[:n] = self._arrays[name][:n]
            new[name] = arr
        self._arrays = new

    def _view(self, name):
        return self._arrays[name][: self._n]

    # ------------------------------------------------------------------
    # Microclusters
    # ------------------------------------------------------------------
    def _effective_limit(self, X):
        """variance_limit, optionally expressed as a fraction of the stream's global variance."""
        shrink = 1.0
        if self.shrink_exp > 0:
            shrink = (max(self.total_num_samples + len(X), 100) / 100.0) ** (
                -self.shrink_exp
            )
        if not self.relative_limit:
            return self.variance_limit * shrink
        if self._g_sum is None:
            self._g_sum = np.zeros(X.shape[1])
        self._g_count += len(X)
        self._g_sum += X.sum(axis=0)
        self._g_sq += float(np.einsum("ij,ij->", X, X))
        g_mean = self._g_sum / self._g_count
        g_var = max(self._g_sq / self._g_count - float(g_mean @ g_mean), EPS)
        scale = (2.0 / X.shape[1]) ** 2 if self.dim_penalty else 1.0
        return self.variance_limit * g_var * scale * shrink

    def _fit_chunk(self, X):
        a = self._arrays
        var_scale = -1.0 if self.dim_penalty else 1.0
        self._n, self._clock = _fit_kernel(
            X,
            a["num_samples"],
            a["mass"],
            a["mean"],
            a["variance"],
            a["density"],
            a["life"],
            a["active"],
            a["last_seen"],
            a["label"],
            self._n,
            self._clock,
            self._effective_limit(X),
            self.fading_factor,
            var_scale,
            self.m_mode,
            self.m_up,
            self.m_a,
            self.m_b,
            self.s_max,
            self.cap_variance,
            self.life_mode,
            self.prune_skip,
            self.born_active,
            float(X.shape[1]) if self.chi_dim else 0.0,
            self.single_assign,
        )
        self.total_num_samples += len(X)

    def fit(self, X, update_macro_clusters=True):
        X = np.ascontiguousarray(X, dtype=float)
        if len(X) == 0:
            return
        self._reserve(len(X), X.shape[1])
        if self.macro_period <= 0:
            self._fit_chunk(X)
            if update_macro_clusters:
                self.update_macro_clusters()
            return
        start = 0
        while start < len(X):
            room = self.macro_period - (self.total_num_samples % self.macro_period)
            stop = min(len(X), start + room)
            self._fit_chunk(X[start:stop])
            if self.total_num_samples % self.macro_period == 0:
                self.update_macro_clusters()
            start = stop

    # ------------------------------------------------------------------
    # Macroclusters
    # ------------------------------------------------------------------
    def _intrinsic_density(self):
        mass, var = self._view("mass"), self._view("variance")
        if self.density == "last":
            return self._view("density").copy()
        if (
            self.density == "peak"
        ):  # density of the paper evaluated at the microcluster's own mean: 1/zeta(mu) = 2S
            return np.where(mass >= 3, 2.0 * self._view("num_samples"), 0.0)
        if self.density == "snapshot":
            return np.where(mass >= 3, mass / (var + self.epsilon), 0.0)
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

        mean, var, mass = self._view("mean"), self._view("variance"), self._view("mass")
        std = np.sqrt(var)
        density = self._intrinsic_density()
        mature = mass > 2

        adjacency = self._adjacency(mean, std)
        _, labels = connected_components(csr_matrix(adjacency), directed=False)

        if self.activation == "none":
            active = mature.copy()
        else:
            if self.activation_scope == "mature":
                count = np.bincount(labels, weights=mature.astype(float))
                total = np.bincount(labels, weights=np.where(mature, density, 0.0))
            else:
                count = np.bincount(labels).astype(float)
                total = np.bincount(labels, weights=density)
            mean_density = total / np.maximum(count, 1.0)
            active = mature & (density >= self.activation_factor * mean_density[labels])
        self._arrays["active"][:n] = active

        idx = np.flatnonzero(active)
        if len(idx) == 0:
            return

        if self.topology == "g1":
            n_macro, macro = connected_components(
                csr_matrix(adjacency[np.ix_(idx, idx)]), directed=False
            )
        else:
            _, macro = np.unique(labels[idx], return_inverse=True)
            n_macro = int(macro.max()) + 1

        self.n_macro_ = n_macro
        macro_ids = np.arange(n_macro)
        if self.stable_labels:
            macro_ids = self._persistent_ids(idx, macro, n_macro, mass)
        indicator = np.zeros((len(idx), n_macro))
        indicator[np.arange(len(idx)), macro] = 1.0
        self._pred = {
            "index": idx,
            "mean": mean[idx].copy(),
            "s": self._view("num_samples")[idx].copy(),
            "mass": mass[idx].copy(),
            "var": var[idx].copy(),
            "density": density[idx].copy(),
            "macro": macro,
            "macro_ids": macro_ids,
            "indicator": indicator,
        }

    def _persistent_ids(self, idx, macro, n_macro, mass):
        """Give each macrocluster the identity that owns most of its mass; unmatched ones get a new one."""
        label = self._arrays["label"]
        old = label[idx]
        ids = np.full(n_macro, -1, dtype=np.int64)
        known = old >= 0
        if known.any():
            pairs = {}
            for c, lab, w in zip(macro[known], old[known], mass[idx][known]):
                pairs[(c, lab)] = pairs.get((c, lab), 0.0) + w
            taken = set()
            for (c, lab), _ in sorted(pairs.items(), key=lambda kv: -kv[1]):
                if ids[c] < 0 and lab not in taken:
                    ids[c] = lab
                    taken.add(lab)
        for c in range(n_macro):
            if ids[c] < 0:
                ids[c] = self._next_label
                self._next_label += 1
        label[idx] = ids[macro]
        return ids

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def _scaled_sq_dist(self, X, mean):
        sq = cdist(X, mean, "sqeuclidean")
        if self.dim_penalty:
            return (np.sqrt(sq) * (2.0 / X.shape[1])) ** 2
        return sq

    def _std_sq(self, X, p):
        """Standardized squared distance d^2 / sigma^2 (inf for degenerate microclusters unless d = 0)."""
        scaled = self._scaled_sq_dist(X, p["mean"])
        positive = p["var"] > 0
        std_sq = np.where(scaled > 0, np.inf, 0.0)
        std_sq[:, positive] = scaled[:, positive] / p["var"][positive]
        return std_sq

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        p = self._pred
        if p is None:
            self.labels_ = np.full(len(X), -1, dtype=int)
            return self.labels_

        s, var, ind = p["s"], p["var"], p["indicator"]
        positive = var > 0
        std_sq = self._std_sq(X, p)
        ecc = 1.0 / s + np.where(positive, std_sq, 0.0) / s

        mode = self.membership
        if mode == "reference":
            weights = ind * (p["density"] / (ind.T @ p["density"])[p["macro"]])[:, None]
            scores = (1.0 - ecc / 2.0) @ weights
        elif mode == "norm_typ":
            weights = ind * (p["density"] / (ind.T @ p["density"])[p["macro"]])[:, None]
            scores = ((1.0 - ecc) / (s - 2.0)) @ weights
        elif mode == "paper":
            dens_x = 2.0 * s / (1.0 + std_sq)  # 1 / zeta(x)
            scores = (
                (dens_x * ((1.0 - ecc) / (s - 2.0)))
                @ ind
                / np.maximum(dens_x @ ind, EPS)
            )
        elif mode == "density_sum":
            scores = (2.0 * p["mass"] / (1.0 + std_sq)) @ ind
        elif mode == "density_max":
            dens_x = 2.0 * p["mass"] / (1.0 + std_sq)
            scores = np.stack(
                [dens_x[:, p["macro"] == c].max(axis=1) for c in range(ind.shape[1])],
                axis=1,
            )
        elif mode == "nearest":
            kern = 1.0 / (1.0 + std_sq)
            scores = np.stack(
                [kern[:, p["macro"] == c].max(axis=1) for c in range(ind.shape[1])],
                axis=1,
            )
        else:
            raise ValueError(mode)

        labels = p["macro_ids"][np.argmax(scores, axis=1)]
        if self.noise_m > 0:
            labels = np.where(std_sq.min(axis=1) > self.noise_m**2, -1, labels)
        self.labels_ = labels
        return labels

    def microcluster_count(self):
        return self._n
