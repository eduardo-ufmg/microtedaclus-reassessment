"""Vectorized, behaviour-preserving reimplementation of the reference MicroTEDAClus.

The microcluster population is stored as a structure of arrays. The sequential
part of the algorithm (recursive update, outlier test, vitality update and
pruning, which depend on the arrival order) runs in a single compiled kernel
over those arrays. The batch parts are vectorized: the macrocluster graph is
rebuilt from a pairwise distance matrix, its components are extracted with
SciPy, and the membership of a whole batch is one matrix product.

No formula of the reference implementation is changed: given the same stream
and hyperparameters, this class reproduces its labels and microcluster counts.
"""

import math

import numpy as np
import optuna
from numba import njit
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import cdist


@njit(cache=True)
def _fit_kernel(
    X,
    num_samples,
    mean,
    variance,
    density,
    life,
    active,
    n,
    variance_limit,
    fading_factor,
    prune,
):
    """Process a batch of samples sequentially; returns the new microcluster count."""
    n_samples, dim = X.shape
    scale = 2.0 / dim
    new_mean = np.empty(dim)

    for k in range(n_samples):
        x = X[k]
        is_new = True

        for i in range(n):
            s = num_samples[i] + 1.0
            ratio = (s - 1.0) / s

            sq = 0.0
            for j in range(dim):
                new_mean[j] = ratio * mean[i, j] + x[j] / s
                delta = x[j] - new_mean[j]
                sq += delta * delta
            scaled = (math.sqrt(sq) * scale) ** 2
            var = ratio * variance[i] + scaled / (s - 1.0)

            if var == 0.0:
                ecc = 1.0 / s
            else:
                ecc = 1.0 / s + scaled / (s * var)
            norm_ecc = ecc / 2.0

            if s < 3.0:
                outlier = var > variance_limit
            else:
                m = 3.0 / (1.0 + math.exp(-0.007 * (s - 100.0)))
                outlier = norm_ecc > (m * m + 1.0) / (2.0 * s)

            if not outlier:
                # Vitality update relative to the state prior to absorbing x
                if variance[i] > 0.0:
                    sq_prev = 0.0
                    for j in range(dim):
                        diff = x[j] - mean[i, j]
                        sq_prev += diff * diff
                    rt = math.sqrt(variance[i])
                    life[i] += ((rt - math.sqrt(sq_prev)) / rt) * fading_factor
                else:
                    life[i] = 1.0

                num_samples[i] = s
                for j in range(dim):
                    mean[i, j] = new_mean[j]
                variance[i] = var
                density[i] = 1.0 / norm_ecc
                is_new = False

        if is_new:
            num_samples[n] = 1.0
            for j in range(dim):
                mean[n, j] = x[j]
            variance[n] = 0.0
            density[n] = 0.0
            life[n] = 1.0
            active[n] = True
            n += 1

        if prune:
            # The reference implementation removes elements from the list it is
            # iterating over, so the microcluster that follows a removed one is
            # skipped in that pass. `skip` reproduces that behaviour.
            m_kept = 0
            skip = False
            for i in range(n):
                keep = True
                if skip:
                    skip = False
                elif not active[i]:
                    life[i] -= fading_factor
                    if life[i] < 0.0:
                        keep = False
                        skip = True
                if keep:
                    if m_kept != i:
                        num_samples[m_kept] = num_samples[i]
                        for j in range(dim):
                            mean[m_kept, j] = mean[i, j]
                        variance[m_kept] = variance[i]
                        density[m_kept] = density[i]
                        life[m_kept] = life[i]
                        active[m_kept] = active[i]
                    m_kept += 1
            n = m_kept

    return n


class MicroTEDAClusVectorized:
    def __init__(self, variance_limit=0.001, decay=100):
        self.variance_limit = variance_limit
        self.fading_factor = 1 / decay
        self.total_num_samples = 0

        self._n = 0
        self._dim = None
        self._num_samples = np.empty(0)
        self._mean = np.empty((0, 0))
        self._variance = np.empty(0)
        self._density = np.empty(0)
        self._life = np.empty(0)
        self._active = np.empty(0, dtype=bool)

        # Prediction model, refreshed by update_macro_clusters
        self._macro_labels = np.empty(0, dtype=int)
        self._n_active_macro = 0
        self._pred = None

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------
    def _reserve(self, extra, dim):
        """Guarantee room for `extra` new microclusters (at most one per sample)."""
        n = self._n
        if self._dim is not None and n + extra <= len(self._num_samples):
            return
        capacity = max(64, 2 * (n + extra))

        def resized(arr, shape, dtype=float):
            new = np.empty(shape, dtype=dtype)
            new[:n] = arr[:n]
            return new

        if self._dim is None:
            self._dim = dim
            self._mean = np.empty((0, dim))
        self._num_samples = resized(self._num_samples, capacity)
        self._mean = resized(self._mean, (capacity, dim))
        self._variance = resized(self._variance, capacity)
        self._density = resized(self._density, capacity)
        self._life = resized(self._life, capacity)
        self._active = resized(self._active, capacity, dtype=bool)

    # ------------------------------------------------------------------
    # Macroclusters
    # ------------------------------------------------------------------
    def update_macro_clusters(self):
        n = self._n
        self._pred = None
        self._n_active_macro = 0
        if n == 0:
            return

        mean = self._mean[:n]
        std = np.sqrt(self._variance[:n])
        density = self._density[:n]

        adjacency = cdist(mean, mean) <= 2.0 * (std[:, None] + std[None, :])
        np.fill_diagonal(adjacency, False)
        _, labels = connected_components(csr_matrix(adjacency), directed=False)
        self._macro_labels = labels

        mean_density = np.bincount(labels, weights=density) / np.bincount(labels)
        active = (self._num_samples[:n] > 2) & (density >= mean_density[labels])
        self._active[:n] = active

        idx = np.flatnonzero(active)
        if len(idx) == 0:
            return

        n_macro, active_labels = connected_components(
            csr_matrix(adjacency[np.ix_(idx, idx)]), directed=False
        )
        total_density = np.bincount(active_labels, weights=density[idx])
        weights = np.zeros((len(idx), n_macro))
        weights[np.arange(len(idx)), active_labels] = (
            density[idx] / total_density[active_labels]
        )

        self._n_active_macro = n_macro
        self._pred = (
            self._mean[idx].copy(),
            self._num_samples[idx].copy(),
            self._variance[idx].copy(),
            weights,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(self, X, update_macro_clusters=True, prune_micro_clusters=True):
        X = np.ascontiguousarray(X, dtype=float)
        if len(X) > 0:
            self._reserve(len(X), X.shape[1])
            self._n = _fit_kernel(
                X,
                self._num_samples,
                self._mean,
                self._variance,
                self._density,
                self._life,
                self._active,
                self._n,
                self.variance_limit,
                self.fading_factor,
                prune_micro_clusters,
            )
            self.total_num_samples += len(X)

        if update_macro_clusters:
            self.update_macro_clusters()

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if self._pred is None:
            self.labels_ = np.full(len(X), -1, dtype=int)
            return self.labels_

        mean, s, variance, weights = self._pred
        scaled = (cdist(X, mean) * (2.0 / X.shape[1])) ** 2
        ecc = np.broadcast_to(1.0 / s, scaled.shape).copy()
        positive = variance > 0
        ecc[:, positive] += scaled[:, positive] / (s[positive] * variance[positive])
        typicality = 1.0 - ecc / 2.0

        self.labels_ = np.argmax(typicality @ weights, axis=1)
        return self.labels_

    def microcluster_count(self):
        return self._n


class MicroTEDAClusVectorizedParameterSearchSpace:
    def sample_parameters(self, trial: optuna.Trial) -> dict:
        return {
            "variance_limit": trial.suggest_float(
                "variance_limit", 0.0001, 100.0, log=True
            ),
            "decay": trial.suggest_int("decay", 10, 10000, log=True),
        }
