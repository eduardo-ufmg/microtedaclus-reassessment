"""Anisotropic variant of the research engine: full covariance and Mahalanobis distance.

Each microcluster keeps its population covariance (same recursion as the scalar
variance) and a shrunk inverse used for the Mahalanobis eccentricity

    xi(x) = 1/S + (x - mu)^T Sigma^-1 (x - mu) / (S d),

which reduces to the Euclidean TEDA eccentricity for an isotropic covariance
(the distance is evaluated against the covariance before the update; see the kernel).
The covariance is shrunk towards the isotropic matrix with the same trace with
intensity alpha = kappa d / (kappa d + S - 1), so young microclusters behave as
the isotropic model and become anisotropic as evidence accumulates.
"""

import math

import numpy as np
from numba import njit
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from investigation.engine import EPS, TedaClus, _m_value


@njit(cache=True)
def _shrunk_inverse(cov, trace, s, dim, kappa, out):
    alpha = kappa * dim / (kappa * dim + s - 1.0)
    iso = max(trace / dim, 1e-300)
    reg = (1.0 - alpha) * cov
    for j in range(dim):
        reg[j, j] += alpha * iso
    out[:, :] = np.linalg.inv(reg)


@njit(cache=True)
def _fit_kernel_cov(
    X,
    num_samples,
    mass,
    mean,
    variance,
    cov,
    cov_inv,
    density,
    life,
    active,
    last_seen,
    label,
    n,
    clock,
    variance_limit,
    fading_factor,
    m_mode,
    m_up,
    m_a,
    m_b,
    s_max,
    cap_variance,
    life_mode,
    born_active,
    chi_dim,
    kappa,
):
    n_samples, dim = X.shape
    v = np.empty(dim)

    for k in range(n_samples):
        x = X[k]
        clock += 1
        is_new = True

        for i in range(n):
            s = num_samples[i] + 1.0
            if s_max > 0.0 and s > s_max:
                s = s_max
            ratio = (s - 1.0) / s

            sq = 0.0
            for j in range(dim):
                v[j] = x[j] - mean[i, j]
                sq += v[j] * v[j]
            # scalar (trace) variance after absorbing x: delta = ratio * v
            scaled = ratio * ratio * sq
            var = ratio * variance[i] + scaled / (s - 1.0)

            if mass[i] + 1.0 < 3.0:
                outlier = var > variance_limit
                norm_ecc = 0.5
            else:
                # Dimension-normalized Mahalanobis distance to the shrunk covariance before the
                # update, q0 = D0^2 / d, mapped through the inclusion identity of the Euclidean
                # recursion, q = c q0 / (1 + c q0 / (S - 1)). For an isotropic covariance this is
                # exactly the reference test. (Including x in the full covariance instead bounds
                # D^2 / d by (S - 1) / d, which cannot exceed m^2 until S > 1 + m^2 d.)
                d0 = 0.0
                for a in range(dim):
                    acc = 0.0
                    for b in range(dim):
                        acc += cov_inv[i, a, b] * v[b]
                    d0 += v[a] * acc
                q0 = max(d0, 0.0) / dim
                dm = ratio * q0 / (1.0 + ratio * q0 / (s - 1.0))
                norm_ecc = (1.0 / s + dm / s) / 2.0
                m = _m_value(s, m_mode, m_up, m_a, m_b)
                if chi_dim > 0.0:
                    limit = (2.0 + m * math.sqrt(2.0 / chi_dim)) / (2.0 * s)
                else:
                    limit = (m * m + 1.0) / (2.0 * s)
                outlier = norm_ecc > limit
                if cap_variance and var > variance_limit:
                    outlier = True

            if not outlier:
                if life_mode == 0:
                    # reference vitality rule, on the scalar (trace) variance
                    if variance[i] > 0.0:
                        rt = math.sqrt(variance[i])
                        life[i] += ((rt - math.sqrt(sq)) / rt) * fading_factor
                    else:
                        life[i] = 1.0
                else:
                    life[i] = 1.0
                coef = ratio * ratio / (s - 1.0)
                for a in range(dim):
                    for b in range(dim):
                        cov[i, a, b] = ratio * cov[i, a, b] + coef * v[a] * v[b]
                for j in range(dim):
                    mean[i, j] += v[j] / s
                num_samples[i] = s
                mass[i] += 1.0
                variance[i] = var
                density[i] = 1.0 / norm_ecc
                last_seen[i] = clock
                _shrunk_inverse(cov[i], var, s, dim, kappa, cov_inv[i])
                is_new = False

        if is_new:
            num_samples[n] = 1.0
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

        m_kept = 0
        for i in range(n):
            keep = True
            if life_mode == 0:
                if not active[i]:
                    life[i] -= fading_factor
                    keep = life[i] >= 0.0
            elif life_mode == 1:
                if last_seen[i] != clock:
                    life[i] -= fading_factor
                    keep = life[i] >= 0.0
            elif life_mode == 2:
                if (not active[i]) and last_seen[i] != clock:
                    life[i] -= fading_factor
                    keep = life[i] >= 0.0
            if keep:
                if m_kept != i:
                    num_samples[m_kept] = num_samples[i]
                    mass[m_kept] = mass[i]
                    mean[m_kept, :] = mean[i, :]
                    variance[m_kept] = variance[i]
                    cov[m_kept, :, :] = cov[i, :, :]
                    cov_inv[m_kept, :, :] = cov_inv[i, :, :]
                    density[m_kept] = density[i]
                    life[m_kept] = life[i]
                    active[m_kept] = active[i]
                    last_seen[m_kept] = last_seen[i]
                    label[m_kept] = label[i]
                m_kept += 1
        n = m_kept

    return n, clock


class TedaCov(TedaClus):
    """TedaClus with full-covariance microclusters (always without the dimensionality penalty)."""

    _FIELDS = TedaClus._FIELDS + ("cov", "cov_inv")

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
        options["prune_skip"] = False
        super().__init__(variance_limit, decay, **options)
        self.kappa = float(kappa)
        self.link_mode = link_mode

    def _reserve(self, extra, dim):
        n = self._n
        if self._dim is not None and n + extra <= len(self._arrays["mass"]):
            return
        old = self._arrays
        self._arrays = {k: v for k, v in old.items() if k not in ("cov", "cov_inv")}
        if self._dim is None:
            self._arrays = {}
        TedaClus._reserve(self, extra, dim)
        capacity = len(self._arrays["mass"])
        for name in ("cov", "cov_inv"):
            arr = np.zeros((capacity, dim, dim))
            if name in old:
                arr[:n] = old[name][:n]
            self._arrays[name] = arr

    def _fit_chunk(self, X):
        a = self._arrays
        self._n, self._clock = _fit_kernel_cov(
            X,
            a["num_samples"],
            a["mass"],
            a["mean"],
            a["variance"],
            a["cov"],
            a["cov_inv"],
            a["density"],
            a["life"],
            a["active"],
            a["last_seen"],
            a["label"],
            self._n,
            self._clock,
            self._effective_limit(X),
            self.fading_factor,
            self.m_mode,
            self.m_up,
            self.m_a,
            self.m_b,
            self.s_max,
            self.cap_variance,
            self.life_mode,
            self.born_active,
            float(X.shape[1]) if self.chi_dim else 0.0,
            self.kappa,
        )
        self.total_num_samples += len(X)

    # ------------------------------------------------------------------
    def _shrunk_cov(self):
        cov, var, s = (
            self._view("cov"),
            self._view("variance"),
            self._view("num_samples"),
        )
        dim = self._dim
        alpha = self.kappa * dim / (self.kappa * dim + np.maximum(s, 1.0) - 1.0)
        return (1.0 - alpha)[:, None, None] * cov + (alpha * var / dim)[
            :, None, None
        ] * np.eye(dim)

    def _adjacency(self, mean, std):
        n, dim = mean.shape
        diff = mean[:, None, :] - mean[None, :, :]
        dist_sq = np.einsum("ijk,ijk->ij", diff, diff)
        shr = self._shrunk_cov()
        if self.link_mode == "directional" or dim > 16:
            # std of each microcluster along the direction that joins the two means,
            # scaled by sqrt(d) so an isotropic covariance recovers dist <= link (sigma_i + sigma_j).
            # (The Bhattacharyya rule needs a d x d solve per pair and is only used for d <= 16.)
            proj = np.einsum(
                "ijk,ijk->ij", np.matmul(diff, shr), diff
            )  # dist^2 * sigma_dir_i^2
            reach = np.sqrt(dim) * (
                np.sqrt(np.maximum(proj, 0.0)) + np.sqrt(np.maximum(proj.T, 0.0))
            )
            adjacency = dist_sq <= self.link * reach
        elif self.link_mode == "bhattacharyya":
            pooled = 0.5 * (shr[:, None] + shr[None, :]) + EPS * np.eye(dim)
            sol = np.linalg.solve(pooled, diff[..., None])[..., 0]
            maha = np.einsum("ijk,ijk->ij", diff, sol)
            _, logdet = np.linalg.slogdet(shr + EPS * np.eye(dim))
            _, logdet_pooled = np.linalg.slogdet(pooled)
            bhat = maha / 8.0 + 0.5 * (
                logdet_pooled - 0.5 * (logdet[:, None] + logdet[None, :])
            )
            # equal isotropic microclusters: D_B = dist^2 / (8 s^2); the reference rule
            # dist <= 2 (sigma_i + sigma_j) corresponds to D_B <= link^2 d / 2
            adjacency = bhat <= self.link**2 * dim / 2.0
        else:
            raise ValueError(self.link_mode)
        np.fill_diagonal(adjacency, False)
        return adjacency

    def update_macro_clusters(self):
        super().update_macro_clusters()
        if self._pred is not None:
            self._pred["cov_inv"] = self._view("cov_inv")[self._pred["index"]].copy()

    def _std_sq(self, X, p):
        diff = X[:, None, :] - p["mean"][None, :, :]
        maha = np.einsum("bak,akl,bal->ba", diff, p["cov_inv"], diff)
        return np.maximum(maha, 0.0) / X.shape[1]
