"""Streams of moving Gaussian clusters, and the two benchmarks built on them.

Port of `DSD_MG` and `DSD_Benchmark` (R package stream). A stream is a set of
components, each with a density in points per unit of time. Time advances by
one unit for every `sum of densities` points, and all the points of a unit of
time are drawn at its start.
"""

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Keyframe:
    time: float
    density: float
    spread: float  # standard deviation of the Gaussian
    center: tuple
    reset: bool = False


class LinearCluster:
    """Gaussian cluster whose attributes are interpolated linearly between keyframes.

    If a keyframe has `reset`, the movement is periodic with the time of that
    keyframe as its period.
    """

    def __init__(self, *keyframes: Keyframe):
        self.keyframes = sorted(keyframes, key=lambda k: k.time)
        resets = [k.time for k in self.keyframes if k.reset]
        self.cycle = resets[0] if resets else None
        self.dimension = len(self.keyframes[0].center)

    def _attributes(self, time):
        t = time % self.cycle + 1 if self.cycle else time
        current = sum(k.time <= t for k in self.keyframes)
        if current == 0:
            return 0.0, None, None
        if current == len(self.keyframes):
            last = self.keyframes[-1]
            return last.density, np.asarray(last.center, dtype=float), last.spread
        a, b = self.keyframes[current - 1], self.keyframes[current]
        w = (t - a.time) / (b.time - a.time)
        return (
            (1 - w) * a.density + w * b.density,
            (1 - w) * np.asarray(a.center, dtype=float) + w * np.asarray(b.center),
            (1 - w) * a.spread + w * b.spread,
        )

    def density(self, time):
        return self._attributes(time)[0]

    def sample(self, time, count, rng):
        _, center, spread = self._attributes(time)
        return rng.normal(center, spread, size=(count, len(center)))


class StaticCluster:
    """Gaussian cluster that does not move."""

    def __init__(self, density, center, spread):
        self._density, self.center, self.spread = (
            density,
            np.asarray(center, float),
            spread,
        )
        self.dimension = len(self.center)

    def density(self, time):
        return self._density

    def sample(self, time, count, rng):
        return rng.normal(self.center, self.spread, size=(count, len(self.center)))


class UniformNoise:
    """Uniform noise in a box."""

    def __init__(self, density, low, high):
        self._density, self.low, self.high = (
            density,
            np.asarray(low, float),
            np.asarray(high, float),
        )
        self.dimension = len(self.low)

    def density(self, time):
        return self._density

    def sample(self, time, count, rng):
        return rng.uniform(self.low, self.high, size=(count, len(self.low)))


def moving_gaussians(n, rng, components, labels):
    """n samples of a stream of components; `labels` gives the label of each of them."""
    dimension = components[0].dimension
    X = np.empty((n, dimension))
    y = np.empty(n, dtype=int)
    labels = np.asarray(labels)
    time, produced = 1.0, 0
    while produced < n:
        density = np.array([component.density(time) for component in components])
        rate = density.sum()
        if rate == 0:
            raise ValueError("no component produces points at this time")
        left_in_unit = rate - (time - math.floor(time)) * rate
        if left_in_unit < 1:
            time = math.ceil(time)
            continue
        count = math.floor(min(left_in_unit, n - produced))
        if count >= 1:
            chosen = rng.choice(len(components), size=count, p=density / rate)
            rows = slice(produced, produced + count)
            block = np.empty((count, dimension))
            for index, component in enumerate(components):
                mask = chosen == index
                if mask.any():
                    block[mask] = component.sample(time, int(mask.sum()), rng)
            X[rows], y[rows] = block, labels[chosen]
        time += count / rate
        produced += count
    return X, y


def benchmark_1(n, rng):
    """Two tight clusters that move along the diagonals, cross at the centre and return, with noise."""
    components = [
        LinearCluster(
            Keyframe(0, 10, 0.01, (0.1, 0.9)),
            Keyframe(250, 10, 0.01, (0.9, 0.1)),
            Keyframe(500, 10, 0.01, (0.1, 0.9), reset=True),
        ),
        LinearCluster(
            Keyframe(0, 10, 0.01, (0.1, 0.1)),
            Keyframe(250, 10, 0.01, (0.9, 0.9)),
            Keyframe(500, 10, 0.01, (0.1, 0.1), reset=True),
        ),
        UniformNoise(2, (0, 0), (1, 1)),
    ]
    return moving_gaussians(n, rng, components, labels=[0, 1, -1])


def benchmark_2(n, rng):
    """Two static clusters in opposite corners and a third that moves between them."""
    components = [
        StaticCluster(1, (0, 0), 0.1),
        StaticCluster(1, (1, 1), 0.1),
        LinearCluster(
            Keyframe(0, 1, 0.1, (0, 0)),
            Keyframe(500, 1, 0.1, (1, 1)),
            Keyframe(1000, 1, 0.1, (0, 0), reset=True),
        ),
    ]
    return moving_gaussians(n, rng, components, labels=[0, 1, 2])
