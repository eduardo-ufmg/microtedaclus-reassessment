"""Axis-aligned anisotropic variant of the engine: one variance per feature.

Each microcluster keeps the variance of each feature, updated by the diagonal of
the recursion of the covariance engine, so their sum is the scalar variance of
the isotropic engine. The variances are shrunk towards the isotropic value, the
scalar variance spread evenly over the d features, with the intensity of the
covariance engine, alpha = kappa d / (kappa d + S - 1): a young microcluster
behaves as in the isotropic engine. This engine and the covariance engine differ
only in the covariances between features, which this one takes as zero; it
stores d values per microcluster instead of d^2.

The outlier test, the intersection of two microclusters and the distance of a
prediction are those of the covariance engine with a diagonal covariance: the
standardized distance is sum_k (x_k - mu_k)^2 / s_k^2 / d, measured before the
sample is absorbed, and two microclusters intersect along the line that joins
their means. Both reduce to the isotropic engine when every feature has the same
variance.
"""

import math

import numpy as np
from numba import njit

from benchmark.improvement.engine import EPS, MicroTEDAClusEngine, _sensitivity


@njit(cache=True)
def _evaluate_diag(i, x, count, mass, mean, variance, fvar, v, variance_limit, m_mode, m_up, m_a, m_b, s_max, cap_variance, chi_dim, kappa,):  # fmt: skip
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

    alpha = kappa * dim / (kappa * dim + count[i] - 1.0)
    iso = max(variance[i] / dim, 1e-300)
    d0 = 0.0
    for j in range(dim):
        shrunk = (1.0 - alpha) * fvar[i, j] + alpha * iso
        d0 += v[j] * v[j] / max(shrunk, 1e-300)
    q0 = d0 / dim
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
def _absorb_diag(i, v, s, var, norm_ecc, sq, clock, count, mass, mean, variance, fvar, density, life, last_seen, fading_factor, life_mode,):  # fmt: skip
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
    for j in range(dim):
        fvar[i, j] = ratio * fvar[i, j] + coef * v[j] * v[j]
        mean[i, j] += v[j] / s
    count[i] = s
    mass[i] += 1.0
    variance[i] = var
    density[i] = 1.0 / norm_ecc
    last_seen[i] = clock


@njit(cache=True)
def _fit_kernel_diag(X, count, mass, mean, variance, fvar, density, life, active, last_seen, label, n, clock, variance_limit, fading_factor, m_mode, m_up, m_a, m_b, s_max, cap_variance, life_mode, born_active, chi_dim, single_assign, kappa,):  # fmt: skip
    n_samples, dim = X.shape
    v = np.empty(dim)

    for k in range(n_samples):
        x = X[k]
        clock += 1
        is_new = True
        best, best_dist = -1, np.inf

        for i in range(n):
            outlier, s, var, norm_ecc, sq = _evaluate_diag(
                i, x, count, mass, mean, variance, fvar, v,
                variance_limit, m_mode, m_up, m_a, m_b, s_max, cap_variance, chi_dim, kappa,
            )  # fmt: skip
            if outlier:
                continue
            is_new = False
            if single_assign:
                if sq < best_dist:
                    best, best_dist = i, sq
            else:
                _absorb_diag(
                    i, v, s, var, norm_ecc, sq, clock, count, mass, mean, variance, fvar,
                    density, life, last_seen, fading_factor, life_mode,
                )  # fmt: skip

        if single_assign and best >= 0:
            outlier, s, var, norm_ecc, sq = _evaluate_diag(
                best, x, count, mass, mean, variance, fvar, v,
                variance_limit, m_mode, m_up, m_a, m_b, s_max, cap_variance, chi_dim, kappa,
            )  # fmt: skip
            _absorb_diag(
                best, v, s, var, norm_ecc, sq, clock, count, mass, mean, variance, fvar,
                density, life, last_seen, fading_factor, life_mode,
            )  # fmt: skip

        if is_new:
            count[n] = 1.0
            mass[n] = 1.0
            for j in range(dim):
                mean[n, j] = x[j]
                fvar[n, j] = 0.0
            variance[n] = 0.0
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
                fvar[kept, :] = fvar[i, :]
                density[kept] = density[i]
                life[kept] = life[i]
                active[kept] = active[i]
                last_seen[kept] = last_seen[i]
                label[kept] = label[i]
            kept += 1
        n = kept

    return n, clock


class MicroTEDAClusDiagonalEngine(MicroTEDAClusEngine):
    """Engine with one variance per feature (always without the dimensionality penalty).

    kappa  strength of the shrinkage towards the isotropic variance
    """

    _FIELDS = MicroTEDAClusEngine._FIELDS + ("fvar",)

    def __init__(self, variance_limit=0.001, decay=100, *, kappa=1.0, **options):
        options["dim_penalty"] = False
        super().__init__(variance_limit, decay, **options)
        self.kappa = float(kappa)

    def _field_shape(self, name, capacity):
        if name == "fvar":
            return (capacity, self._dim)
        return super()._field_shape(name, capacity)

    def _fit_batch(self, X):
        a = self._arrays
        self._n, self._clock = _fit_kernel_diag(
            X, a["count"], a["mass"], a["mean"], a["variance"], a["fvar"],
            a["density"], a["life"], a["active"], a["last_seen"], a["label"], self._n, self._clock,
            self._current_limit(X), self.fading_factor, self.m_mode, self.m_up, self.m_a, self.m_b,
            self.s_max, self.cap_variance, self.life_mode, self.born_active,
            float(X.shape[1]) if self.chi_dim else 0.0, self.single_assign, self.kappa,
        )  # fmt: skip

    def _shrunk_fvar(self):
        fvar, variance, s = (
            self._view("fvar"),
            self._view("variance"),
            self._view("count"),
        )
        dim = self._dim
        alpha = self.kappa * dim / (self.kappa * dim + np.maximum(s, 1.0) - 1.0)
        return (1.0 - alpha)[:, None] * fvar + (alpha * variance / dim)[:, None]

    def _log_volume(self):
        return 0.5 * np.log(np.maximum(self._shrunk_fvar(), EPS)).sum(axis=1)

    def _adjacency(self, mean, std):
        """Intersection along the line that joins the means: dist <= link sqrt(d) (s_i + s_j)."""
        dim = mean.shape[1]
        shrunk = self._shrunk_fvar()
        squares = mean * mean
        dist_sq = np.maximum(
            squares.sum(axis=1)[:, None]
            + squares.sum(axis=1)[None, :]
            - 2.0 * mean @ mean.T,
            0.0,
        )
        # projected[i, j] = sum_k (mu_ik - mu_jk)^2 s_ik^2 = dist^2 s_i^2 along the line
        projected = np.maximum(
            (squares * shrunk).sum(axis=1)[:, None]
            - 2.0 * (mean * shrunk) @ mean.T
            + shrunk @ squares.T,
            0.0,
        )
        reach = np.sqrt(dim) * (np.sqrt(projected) + np.sqrt(projected.T))
        adjacency = dist_sq <= self.link * reach
        np.fill_diagonal(adjacency, False)
        return adjacency

    def update_macro_clusters(self):
        super().update_macro_clusters()
        if self._pred is not None:
            shrunk = self._shrunk_fvar()[self._pred["index"]]
            self._pred["inv_fvar"] = 1.0 / np.maximum(shrunk, 1e-300)

    def _standardized_sq_distance(self, X, p):
        inv = p["inv_fvar"]
        distance = (
            (X * X) @ inv.T
            - 2.0 * X @ (p["mean"] * inv).T
            + (p["mean"] ** 2 * inv).sum(axis=1)[None, :]
        )
        return np.maximum(distance, 0.0) / X.shape[1]
