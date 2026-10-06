"""Anisotropic variant of the engine: full covariance and Mahalanobis distance.

Each microcluster keeps its population covariance, updated by the same recursion
as the scalar variance, and the inverse of a shrunk version of it. The shrinkage
pulls the covariance towards the isotropic matrix with the same trace, with
intensity alpha = kappa d / (kappa d + S - 1), so a young microcluster behaves as
in the isotropic engine and becomes anisotropic as evidence accumulates.

The outlier test replaces the standardized Euclidean distance d^2 / sigma^2 by
the dimension-normalized Mahalanobis distance D^2 / d, which coincides with it
for an isotropic covariance. The distance is measured against the covariance
before the sample is absorbed: including the sample in a full covariance bounds
D^2 / d by (S - 1) / d, which cannot exceed m^2 until S > 1 + m^2 d.
"""

import math

import numpy as np
from numba import njit

from model_improvement.engine import EPS, MicroTEDAClusEngine, _sensitivity


@njit(cache=True)
def _shrunk_inverse(cov, trace, s, dim, kappa, out):
    alpha = kappa * dim / (kappa * dim + s - 1.0)
    iso = max(trace / dim, 1e-300)
    reg = (1.0 - alpha) * cov
    for j in range(dim):
        reg[j, j] += alpha * iso
    out[:, :] = np.linalg.inv(reg)


@njit(cache=True)
def _evaluate_cov(i, x, count, mass, mean, variance, cov_inv, v, variance_limit, m_mode, m_up, m_a, m_b, s_max, cap_variance, chi_dim,):  # fmt: skip
    """Tentatively absorb x into microcluster i; returns (outlier, s, var, norm_ecc, sq).

    v receives x - mean and sq its squared norm.
    """
    dim = len(x)
    s = count[i] + 1.0
    if s_max > 0.0 and s > s_max:
        s = s_max
    ratio = (s - 1.0) / s

    sq = 0.0
    for j in range(dim):
        v[j] = x[j] - mean[i, j]
        sq += v[j] * v[j]
    # trace of the covariance after absorbing x (x - new mean = ratio * v)
    var = ratio * variance[i] + ratio * ratio * sq / (s - 1.0)

    if mass[i] + 1.0 < 3.0:
        return var > variance_limit, s, var, 0.5, sq

    d0 = 0.0
    for a in range(dim):
        acc = 0.0
        for b in range(dim):
            acc += cov_inv[i, a, b] * v[b]
        d0 += v[a] * acc
    q0 = max(d0, 0.0) / dim
    # inclusion identity of the scalar recursion applied to q0
    q = ratio * q0 / (1.0 + ratio * q0 / (s - 1.0))
    norm_ecc = (1.0 / s + q / s) / 2.0
    m = _sensitivity(s, m_mode, m_up, m_a, m_b)
    if chi_dim > 0.0:
        limit = (2.0 + m * math.sqrt(2.0 / chi_dim)) / (2.0 * s)
    else:
        limit = (m * m + 1.0) / (2.0 * s)
    outlier = norm_ecc > limit
    if cap_variance and var > variance_limit:
        outlier = True
    return outlier, s, var, norm_ecc, sq


@njit(cache=True)
def _absorb_cov(i, v, s, var, norm_ecc, sq, clock, count, mass, mean, variance, cov, cov_inv, density, life, last_seen, fading_factor, life_mode, kappa,):  # fmt: skip
    dim = len(v)
    if life_mode == 0:
        if variance[i] > 0.0:
            radius = math.sqrt(variance[i])
            life[i] += ((radius - math.sqrt(sq)) / radius) * fading_factor
        else:
            life[i] = 1.0
    else:
        life[i] = 1.0

    ratio = (s - 1.0) / s
    coef = ratio * ratio / (s - 1.0)
    for a in range(dim):
        for b in range(dim):
            cov[i, a, b] = ratio * cov[i, a, b] + coef * v[a] * v[b]
    for j in range(dim):
        mean[i, j] += v[j] / s
    count[i] = s
    mass[i] += 1.0
    variance[i] = var
    density[i] = 1.0 / norm_ecc
    last_seen[i] = clock
    _shrunk_inverse(cov[i], var, s, dim, kappa, cov_inv[i])


@njit(cache=True)
def _fit_kernel_cov(X, count, mass, mean, variance, cov, cov_inv, density, life, active, last_seen, label, n, clock, variance_limit, fading_factor, m_mode, m_up, m_a, m_b, s_max, cap_variance, life_mode, born_active, chi_dim, single_assign, kappa,):  # fmt: skip
    n_samples, dim = X.shape
    v = np.empty(dim)

    for k in range(n_samples):
        x = X[k]
        clock += 1
        is_new = True
        best, best_dist = -1, np.inf

        for i in range(n):
            outlier, s, var, norm_ecc, sq = _evaluate_cov(
                i, x, count, mass, mean, variance, cov_inv, v,
                variance_limit, m_mode, m_up, m_a, m_b, s_max, cap_variance, chi_dim,
            )  # fmt: skip
            if outlier:
                continue
            is_new = False
            if single_assign:
                if sq < best_dist:
                    best, best_dist = i, sq
            else:
                _absorb_cov(
                    i, v, s, var, norm_ecc, sq, clock, count, mass, mean, variance, cov, cov_inv,
                    density, life, last_seen, fading_factor, life_mode, kappa,
                )  # fmt: skip

        if single_assign and best >= 0:
            outlier, s, var, norm_ecc, sq = _evaluate_cov(
                best, x, count, mass, mean, variance, cov_inv, v,
                variance_limit, m_mode, m_up, m_a, m_b, s_max, cap_variance, chi_dim,
            )  # fmt: skip
            _absorb_cov(
                best, v, s, var, norm_ecc, sq, clock, count, mass, mean, variance, cov, cov_inv,
                density, life, last_seen, fading_factor, life_mode, kappa,
            )  # fmt: skip

        if is_new:
            count[n] = 1.0
            mass[n] = 1.0
            for j in range(dim):
                mean[n, j] = x[j]
            variance[n] = 0.0
            cov[n, :, :] = 0.0
            cov_inv[n, :, :] = 0.0
            density[n] = 0.0
            life[n] = 1.0
            active[n] = born_active
            last_seen[n] = clock
            label[n] = -1
            n += 1

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
                mean[kept, :] = mean[i, :]
                variance[kept] = variance[i]
                cov[kept, :, :] = cov[i, :, :]
                cov_inv[kept, :, :] = cov_inv[i, :, :]
                density[kept] = density[i]
                life[kept] = life[i]
                active[kept] = active[i]
                last_seen[kept] = last_seen[i]
                label[kept] = label[i]
            kept += 1
        n = kept

    return n, clock


class MicroTEDAClusCovarianceEngine(MicroTEDAClusEngine):
    """Engine with full-covariance microclusters (always without the dimensionality penalty).

    link_mode  how two microclusters intersect:
               directional    dist <= link sqrt(d) (s_i + s_j), with s the standard deviation of
                              each microcluster along the line that joins the two means
               bhattacharyya  Bhattacharyya distance <= link^2 d / 2 (for d <= 16; directional above)
               Both reduce to dist <= link (sigma_i + sigma_j) for equal isotropic microclusters.
    kappa      strength of the shrinkage towards an isotropic covariance
    """

    _FIELDS = MicroTEDAClusEngine._FIELDS + ("cov", "cov_inv")

    def __init__(
        self,
        variance_limit=0.001,
        decay=100,
        *,
        kappa=1.0,
        link_mode="directional",
        **options
    ):
        options["dim_penalty"] = False
        super().__init__(variance_limit, decay, **options)
        self.kappa = float(kappa)
        self.link_mode = link_mode

    def _field_shape(self, name, capacity):
        if name in ("cov", "cov_inv"):
            return (capacity, self._dim, self._dim)
        return super()._field_shape(name, capacity)

    def _fit_batch(self, X):
        a = self._arrays
        self._n, self._clock = _fit_kernel_cov(
            X, a["count"], a["mass"], a["mean"], a["variance"], a["cov"], a["cov_inv"],
            a["density"], a["life"], a["active"], a["last_seen"], a["label"], self._n, self._clock,
            self._current_limit(X), self.fading_factor, self.m_mode, self.m_up, self.m_a, self.m_b,
            self.s_max, self.cap_variance, self.life_mode, self.born_active,
            float(X.shape[1]) if self.chi_dim else 0.0, self.single_assign, self.kappa,
        )  # fmt: skip

    def _shrunk_covariance(self):
        cov, variance, s = (
            self._view("cov"),
            self._view("variance"),
            self._view("count"),
        )
        dim = self._dim
        alpha = self.kappa * dim / (self.kappa * dim + np.maximum(s, 1.0) - 1.0)
        return (1.0 - alpha)[:, None, None] * cov + (alpha * variance / dim)[
            :, None, None
        ] * np.eye(dim)

    def _adjacency(self, mean, std):
        dim = mean.shape[1]
        diff = mean[:, None, :] - mean[None, :, :]
        dist_sq = np.einsum("ijk,ijk->ij", diff, diff)
        shrunk = self._shrunk_covariance()
        if self.link_mode == "directional" or dim > 16:
            projected = np.einsum(
                "ijk,ijk->ij", np.matmul(diff, shrunk), diff
            )  # dist^2 s_i^2
            reach = np.sqrt(dim) * (
                np.sqrt(np.maximum(projected, 0.0))
                + np.sqrt(np.maximum(projected.T, 0.0))
            )
            adjacency = dist_sq <= self.link * reach
        elif self.link_mode == "bhattacharyya":
            pooled = 0.5 * (shrunk[:, None] + shrunk[None, :]) + EPS * np.eye(dim)
            solved = np.linalg.solve(pooled, diff[..., None])[..., 0]
            mahalanobis = np.einsum("ijk,ijk->ij", diff, solved)
            _, logdet = np.linalg.slogdet(shrunk + EPS * np.eye(dim))
            _, logdet_pooled = np.linalg.slogdet(pooled)
            distance = mahalanobis / 8.0 + 0.5 * (
                logdet_pooled - 0.5 * (logdet[:, None] + logdet[None, :])
            )
            adjacency = distance <= self.link**2 * dim / 2.0
        else:
            raise ValueError(self.link_mode)
        np.fill_diagonal(adjacency, False)
        return adjacency

    def update_macro_clusters(self):
        super().update_macro_clusters()
        if self._pred is not None:
            self._pred["cov_inv"] = self._view("cov_inv")[self._pred["index"]].copy()

    def _standardized_sq_distance(self, X, p):
        diff = X[:, None, :] - p["mean"][None, :, :]
        mahalanobis = np.einsum("bak,akl,bal->ba", diff, p["cov_inv"], diff)
        return np.maximum(mahalanobis, 0.0) / X.shape[1]
