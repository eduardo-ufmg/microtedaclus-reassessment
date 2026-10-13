"""Cassini: two banana-shaped clusters around a circular one.

Port of `mlbench.cassini` (R package mlbench): each cluster is sampled by
rejection from its bounding box, with the relative sizes 2:2:1.
"""

import numpy as np

A = 1.0
C = 0.97
C_INNER = np.sqrt((1.0 + C**2) / 3.0)
A_INNER = C_INNER * np.sqrt(2.0)
SHIFT = 1.1
RADIUS = 0.6
HALF_WIDTH = np.sqrt(A**2 + C**2)


def _cassini_oval(x, a, c):
    """Positive branch of the Cassini oval with parameters a and c."""
    return np.sqrt(np.maximum(-(c**2) - x**2 + np.sqrt(a**4 + 4.0 * c**2 * x**2), 0.0))


def _rejection(n, rng, low, high, accept):
    """Draw n points uniformly in the box [low, high] that satisfy `accept`."""
    points = np.empty((0, 2))
    while len(points) < n:
        candidates = rng.uniform(low, high, size=(2 * (n - len(points)) + 16, 2))
        points = np.vstack(
            [points, candidates[accept(candidates[:, 0], candidates[:, 1])]]
        )
    return points[:n]


def cassini(n, rng, relative_sizes=(2, 2, 1)):
    """n samples of the Cassini problem, ordered by cluster."""
    piece = n / sum(relative_sizes)
    n_lower, n_upper, n_circle = (round(size * piece) for size in relative_sizes)
    if n_lower + n_upper + n_circle != n:
        n_circle += 1

    lower = _rejection(
        n_lower,
        rng,
        (-HALF_WIDTH, -SHIFT - 1.0),
        (HALF_WIDTH, -SHIFT + 0.6),
        lambda x, y: (y < _cassini_oval(x, A, C) - SHIFT)
        & (y > -_cassini_oval(x, A_INNER, C_INNER) - SHIFT),
    )
    upper = _rejection(
        n_upper,
        rng,
        (-HALF_WIDTH, SHIFT - 0.6),
        (HALF_WIDTH, SHIFT + 1.0),
        lambda x, y: (y > -_cassini_oval(x, A, C) + SHIFT)
        & (y < _cassini_oval(x, A_INNER, C_INNER) + SHIFT),
    )
    circle = _rejection(
        n_circle,
        rng,
        (-RADIUS, -RADIUS),
        (RADIUS, RADIUS),
        lambda x, y: x**2 + y**2 < RADIUS**2,
    )
    X = np.vstack([lower, upper, circle])
    y = np.repeat([0, 1, 2], [n_lower, n_upper, n_circle])
    return X, y
