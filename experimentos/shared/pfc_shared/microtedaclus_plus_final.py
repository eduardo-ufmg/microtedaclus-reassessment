"""MicroTEDAClus+ in its final version, from the improvement on the real datasets.

The algorithm keeps the structure of MicroTEDAClus: TEDA microclusters updated
recursively, macroclusters as the connected components of the graph of
intersecting active microclusters, and vitality-based pruning of the inactive
ones.

Microclusters
    A sample updates only the nearest microcluster to which it is not an
    outlier, or creates a microcluster if it is an outlier to all of them. A
    microcluster with fewer than three samples accepts a sample while its
    variance stays within `variance_limit`, a fraction of the running variance
    of the stream; from three samples on, the eccentricity test with the
    sensitivity m(S) of MicroTEDAClus decides. Distances are not scaled by the
    dimension.

Macroclusters
    Two microclusters intersect when the distance between their means is at
    most `link` times the sum of their standard deviations. In each connected
    component of the intersection graph, a microcluster with three or more
    samples is active if its density is not below the mean density of the
    component, and the macroclusters are the connected components of the active
    microclusters.

Prediction
    A sample belongs to the macrocluster of the active microcluster to which it
    has the smallest standardized distance d^2 / sigma^2. There is no noise
    label: only the samples predicted before any microcluster is active are
    labelled -1.

The microcluster population is a structure of arrays. The part of the algorithm
that depends on the arrival order runs in one compiled kernel; the macrocluster
graph and the prediction of a batch are array operations.
"""

import math

import numpy as np
from numba import njit
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import cdist


@njit(cache=True)
def _tentative_update(i, x, count, mean, variance, new_mean):
    """Statistics of microcluster i if it absorbed x: (s, var, eccentricity, sq_previous)."""
    dim = len(x)
    s = count[i] + 1.0
    ratio = (s - 1.0) / s

    sq = 0.0
    sq_previous = 0.0
    for j in range(dim):
        new_mean[j] = ratio * mean[i, j] + x[j] / s
        delta = x[j] - new_mean[j]
        sq += delta * delta
        diff = x[j] - mean[i, j]
        sq_previous += diff * diff
    var = ratio * variance[i] + sq / (s - 1.0)

    eccentricity = 1.0 / s
    if var > 0.0:
        eccentricity += sq / (s * var)
    return s, var, eccentricity, sq_previous


@njit(cache=True)
def _update_microclusters(
    X, count, mean, variance, density, vitality, active, n, variance_limit, decay_step
):
    """Process a batch of samples in arrival order; returns the microcluster count."""
    n_samples, dim = X.shape
    new_mean = np.empty(dim)

    for k in range(n_samples):
        x = X[k]

        # the nearest microcluster to which x is not an outlier
        nearest, nearest_sq = -1, np.inf
        for i in range(n):
            s, var, eccentricity, sq_previous = _tentative_update(
                i, x, count, mean, variance, new_mean
            )
            if s < 3.0:
                outlier = var > variance_limit
            else:
                m = 3.0 / (1.0 + math.exp(-0.007 * (s - 100.0)))
                outlier = eccentricity / 2.0 > (m * m + 1.0) / (2.0 * s)
            if not outlier and sq_previous < nearest_sq:
                nearest, nearest_sq = i, sq_previous

        if nearest >= 0:
            i = nearest
            s, var, eccentricity, sq_previous = _tentative_update(
                i, x, count, mean, variance, new_mean
            )
            # vitality grows with the closeness of x to the previous mean
            if variance[i] > 0.0:
                radius = math.sqrt(variance[i])
                vitality[i] += ((radius - math.sqrt(sq_previous)) / radius) * decay_step
            else:
                vitality[i] = 1.0
            count[i] = s
            for j in range(dim):
                mean[i, j] = new_mean[j]
            variance[i] = var
            density[i] = 2.0 / eccentricity  # density of the absorbed sample, 1 / zeta
        else:
            count[n] = 1.0
            for j in range(dim):
                mean[n, j] = x[j]
            variance[n] = 0.0
            density[n] = 0.0
            vitality[n] = 1.0
            active[n] = True
            n += 1

        # inactive microclusters lose vitality and are pruned when it runs out
        kept = 0
        for i in range(n):
            if not active[i]:
                vitality[i] -= decay_step
                if vitality[i] < 0.0:
                    continue
            if kept != i:
                count[kept] = count[i]
                for j in range(dim):
                    mean[kept, j] = mean[i, j]
                variance[kept] = variance[i]
                density[kept] = density[i]
                vitality[kept] = vitality[i]
                active[kept] = active[i]
            kept += 1
        n = kept

    return n


class MicroTEDAClusPlusFinal:
    def __init__(self, variance_limit=0.01, decay=100, link=2.0):
        self.variance_limit = float(variance_limit)
        self.decay_step = 1.0 / decay
        self.link = float(link)
        self.total_num_samples = 0

        self._n = 0
        self._count = np.empty(0)
        self._mean = np.empty((0, 0))
        self._variance = np.empty(0)
        self._density = np.empty(0)
        self._vitality = np.empty(0)
        self._active = np.empty(0, dtype=bool)

        # running statistics of the stream, for the relative variance limit
        self._stream_sum = None
        self._stream_sq = 0.0

        # means, variances and macrocluster of the active microclusters
        self._prediction = None

    # ------------------------------------------------------------------
    # Microclusters
    # ------------------------------------------------------------------
    def _reserve(self, extra, dim):
        """Guarantee room for `extra` new microclusters (at most one per sample)."""
        n = self._n
        if n + extra <= len(self._count):
            return
        capacity = max(64, 2 * (n + extra))

        def grown(array, shape, dtype=float):
            new = np.zeros(shape, dtype=dtype)
            new[:n] = array[:n]
            return new

        self._count = grown(self._count, capacity)
        self._mean = grown(self._mean.reshape(-1, dim), (capacity, dim))
        self._variance = grown(self._variance, capacity)
        self._density = grown(self._density, capacity)
        self._vitality = grown(self._vitality, capacity)
        self._active = grown(self._active, capacity, dtype=bool)

    def _absolute_variance_limit(self, X):
        """variance_limit as a fraction of the variance of the stream seen so far."""
        if self._stream_sum is None:
            self._stream_sum = np.zeros(X.shape[1])
        self._stream_sum += X.sum(axis=0)
        self._stream_sq += float(np.einsum("ij,ij->", X, X))
        seen = self.total_num_samples + len(X)
        stream_mean = self._stream_sum / seen
        stream_variance = self._stream_sq / seen - float(stream_mean @ stream_mean)
        return self.variance_limit * max(stream_variance, 1e-12)

    def fit(self, X):
        X = np.ascontiguousarray(X, dtype=float)
        if len(X) == 0:
            return
        self._reserve(len(X), X.shape[1])
        self._n = _update_microclusters(
            X,
            self._count,
            self._mean,
            self._variance,
            self._density,
            self._vitality,
            self._active,
            self._n,
            self._absolute_variance_limit(X),
            self.decay_step,
        )
        self.total_num_samples += len(X)
        self._update_macroclusters()

    # ------------------------------------------------------------------
    # Macroclusters
    # ------------------------------------------------------------------
    def _update_macroclusters(self):
        n = self._n
        self._prediction = None
        if n == 0:
            return
        count, mean, variance = self._count[:n], self._mean[:n], self._variance[:n]
        density = self._density[:n]

        # microclusters intersect when their means are within `link` times the sum of
        # their standard deviations
        std = np.sqrt(variance)
        intersects = cdist(mean, mean) <= self.link * (std[:, None] + std[None, :])
        np.fill_diagonal(intersects, False)
        _, component = connected_components(csr_matrix(intersects), directed=False)

        # a microcluster with three or more samples is active if it is at least as dense
        # as the average microcluster of its component
        mean_density = np.bincount(component, weights=density) / np.bincount(component)
        active = (count > 2) & (density >= mean_density[component])
        self._active[:n] = active

        index = np.flatnonzero(active)
        if len(index) == 0:
            return

        # macroclusters are the connected components of the active microclusters
        _, macro = connected_components(
            csr_matrix(intersects[np.ix_(index, index)]), directed=False
        )
        self._prediction = (mean[index].copy(), variance[index].copy(), macro)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if self._prediction is None:
            self.labels_ = np.full(len(X), -1, dtype=int)
            return self.labels_
        mean, variance, macro = self._prediction

        # standardized squared distance to every active microcluster
        sq = cdist(X, mean, "sqeuclidean")
        positive = variance > 0
        distance = np.where(sq > 0, np.inf, 0.0)
        distance[:, positive] = sq[:, positive] / variance[positive]

        self.labels_ = macro[np.argmin(distance, axis=1)]
        return self.labels_

    def microcluster_count(self):
        return self._n
