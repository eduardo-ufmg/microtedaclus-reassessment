"""Random RBF stream with cluster events: clusters that move, merge, split, appear and disappear.

Port of `moa.streams.clustering.RandomRBFGeneratorEvents` (MOA), as the R
package streamMOA configures it. Clusters are spheres in the unit hypercube.
Every sample comes from the noise, with probability `noise_level`, or from a
cluster chosen by its weight, at a uniform distance from its centre. Clusters
move in straight lines towards random destinations and, every
`event_frequency` samples, one event is scheduled: a merge, a split, a deletion
or a creation.

The port follows the reference step by step, with two deliberate exceptions.
Noise is labelled -1, while the reference gives it the class value that the
first cluster created after the initial ones also receives. And the options
that the experiments leave at zero, the ranges of the radius, of the weight and
of the speed, are not ported. Noise is uniform in the hypercube and may fall
inside clusters, which is how the reference behaves under streamMOA.
"""

import math
from dataclasses import dataclass

import numpy as np

MERGE_OVERLAP = 0.7  # overlap degree at which two approaching clusters become one
MOVE_EVERY = 10  # clusters move once every this many samples
STEP_LENGTH = 0.01  # distance of the movement scale (see `speed`)
PLACEMENT_ATTEMPTS = 50

MERGE, SPLIT, DELETE, CREATE = 0, 1, 2, 3


@dataclass
class _Kernel:
    label: int
    center: np.ndarray
    radius: float
    weight: float
    remaining: int = -1  # samples until a dying cluster is removed; -1 for a living one
    merging: bool = False
    step: np.ndarray | None = None
    total_steps: int = 0
    steps_done: int = 0


class _Generator:
    def __init__(
        self,
        rng,
        dimension,
        n_clusters,
        cluster_range,
        radius,
        speed,
        noise_level,
        event_frequency,
        merge_split,
        delete_create,
        decay_horizon,
    ):
        self.rng = rng
        self.dimension = dimension
        self.n_clusters = n_clusters
        self.cluster_range = cluster_range
        self.radius = radius
        self.speed = speed
        self.noise_level = noise_level
        self.event_frequency = event_frequency
        self.merge_split = merge_split
        self.delete_create = delete_create
        self.decay_horizon = decay_horizon

        self.kernels: list[_Kernel] = []
        self.active = 0
        self.next_label = 0
        self.merge_pair: tuple[_Kernel, _Kernel] | None = None
        self.countdown = event_frequency
        # The first event is chosen before any cluster exists, so it adds one.
        self.event = self._next_event()
        for _ in range(n_clusters):
            self._create()
        self._normalize()

    # -- clusters ---------------------------------------------------------------
    def _create(self):
        """Add a cluster at a random position inside the hypercube."""
        for _ in range(PLACEMENT_ATTEMPTS):
            center = self.rng.random(self.dimension)
            if np.all(center - self.radius >= 0) and np.all(center + self.radius <= 1):
                kernel = _Kernel(
                    self.next_label, center, self.radius, 1.0 / self.n_clusters
                )
                self.next_label += 1
                self._set_destination(kernel)
                self.kernels.append(kernel)
                self.active += 1
                return
        raise ValueError("no room for a new cluster: reduce the radius")

    def _set_destination(self, kernel, destination=None):
        if destination is None:
            destination = self.rng.random(self.dimension)
        vector = destination - kernel.center
        kernel.total_steps = int(
            np.linalg.norm(vector) / (STEP_LENGTH * MOVE_EVERY) * self.speed
        )
        kernel.step = vector / kernel.total_steps if kernel.total_steps else None
        kernel.steps_done = 0

    def _move(self, kernel):
        if kernel.steps_done >= kernel.total_steps:
            if not kernel.merging:
                self._set_destination(kernel)
            return
        kernel.steps_done += 1
        if kernel.step is None:
            return
        while True:
            center = kernel.center + kernel.step
            if np.all(center - kernel.radius >= 0) and np.all(
                center + kernel.radius <= 1
            ):
                kernel.center = center
                return
            self._set_destination(kernel)
            if kernel.step is None:
                return

    def _normalize(self):
        total = sum(kernel.weight for kernel in self.kernels)
        for kernel in self.kernels:
            kernel.weight /= total

    def _living(self):
        """A living cluster chosen uniformly."""
        while True:
            kernel = self.kernels[self.rng.integers(len(self.kernels))]
            if kernel.remaining == -1:
                return kernel

    def _retire(self, kernel):
        """Stop a cluster from producing samples; it is removed after the decay horizon."""
        kernel.remaining = self.decay_horizon
        kernel.weight = 0.0
        self.active -= 1
        self._normalize()

    # -- events -----------------------------------------------------------------
    def _next_event(self):
        low = self.active <= self.n_clusters - self.cluster_range
        high = self.active >= self.n_clusters + self.cluster_range
        if low and high:
            return None
        if self.merge_split and self.delete_create:
            family = MERGE if self.rng.integers(2) == 0 else DELETE
        elif self.merge_split:
            family = MERGE
        elif self.delete_create:
            family = DELETE
        else:
            return None
        shrink, grow = (MERGE, SPLIT) if family == MERGE else (DELETE, CREATE)
        if low:
            return grow
        if high:
            return shrink
        return shrink if self.rng.integers(2) == 0 else grow

    def _split(self):
        parent = self._living()
        child = _Kernel(
            self.next_label, parent.center.copy(), self.radius, 1.0 / self.n_clusters
        )
        self.next_label += 1
        self._set_destination(child)
        self.kernels.append(child)
        self._normalize()
        self.active += 1

    def _merge(self):
        """Start or advance a merge; returns True when the two clusters have become one."""
        if self.merge_pair is None:
            wanted = 2 * int(self.countdown / self.speed) * STEP_LENGTH
            best = math.inf
            living = [k for k in self.kernels if k.remaining == -1]
            for i, first in enumerate(living):
                for second in living[:i]:
                    gap = np.linalg.norm(first.center - second.center) - wanted
                    if abs(gap) < best and (
                        best != math.inf or gap > 0 or abs(gap) < 0.001
                    ):
                        best = abs(gap)
                        self.merge_pair = (first, second)
            if self.merge_pair is None:
                return False
            first, second = self.merge_pair
            meeting = first.center + 0.5 * (second.center - first.center)
            first.merging = second.merging = True
            self._set_destination(first, meeting)
            self._set_destination(second, meeting)
            return False

        first, second = self.merge_pair
        distance = np.linalg.norm(first.center - second.center)
        small, big = sorted((first.radius, second.radius))
        if distance > small + big:
            overlap = 0.0
        elif distance + small <= big:
            overlap = 1.0
        else:
            overlap = (small + big - distance) / (2 * small)
        if overlap <= MERGE_OVERLAP:
            return False
        # The surviving cluster keeps its centre, as in the reference.
        first.radius = big
        first.weight += second.weight
        first.merging = False
        self._retire(second)
        self.merge_pair = None
        return True

    def _schedule(self):
        """Housekeeping, movement and events that precede every sample."""
        position = 0
        while position < len(self.kernels):
            kernel = self.kernels[position]
            if kernel.remaining == 0:
                self.kernels.remove(kernel)
            elif kernel.remaining > 0:
                kernel.remaining -= 1
            position += 1

        self.countdown -= 1
        if self.countdown % MOVE_EVERY == 0:
            for kernel in list(self.kernels):
                self._move(kernel)
        if self.event_frequency == 0 or self.event is None:
            return

        room_to_shrink = (
            self.active > 1 and self.active > self.n_clusters - self.cluster_range
        )
        room_to_grow = self.active < self.n_clusters + self.cluster_range
        finished = False
        if self.event == MERGE:
            finished = room_to_shrink and self._merge()
        elif self.countdown <= 0:
            if self.event == SPLIT and room_to_grow:
                self._split()
            elif self.event == DELETE and room_to_shrink:
                self._retire(self._living())
            elif self.event == CREATE and room_to_grow:
                self._create()
                self._normalize()
            finished = True
        if finished:
            self.countdown = self.event_frequency
            self.event = self._next_event()

    # -- samples ----------------------------------------------------------------
    def _point_in(self, kernel):
        """A point at a uniform distance from the centre, in a uniform direction of the hyperspherical angles."""
        length = self.rng.random() * kernel.radius
        angles = self.rng.random(self.dimension - 1) * 2 * math.pi
        sines = np.cumprod(np.sin(angles))
        offsets = np.empty(self.dimension)
        offsets[0] = np.cos(angles[0])
        offsets[1:-1] = sines[:-1] * np.cos(angles[1:])
        offsets[-1] = sines[-1]
        return kernel.center + length * offsets

    def sample(self):
        self._schedule()
        if self.rng.random() <= self.noise_level:
            return self.rng.random(self.dimension), -1
        draw = self.rng.random()
        for kernel in self.kernels:
            draw -= kernel.weight
            if draw <= 0:
                return self._point_in(kernel), kernel.label
        return self._point_in(kernel), kernel.label


def rbf_events(
    n,
    rng,
    dimension=2,
    n_clusters=3,
    cluster_range=3,
    radius=0.07,
    speed=100,
    noise_level=0.1,
    event_frequency=1000,
    merge_split=True,
    delete_create=True,
    decay_horizon=1000,
):
    """n samples of the stream; the defaults are the configuration of the experiments."""
    generator = _Generator(
        rng,
        dimension,
        n_clusters,
        cluster_range,
        radius,
        speed,
        noise_level,
        event_frequency,
        merge_split,
        delete_create,
        decay_horizon,
    )
    X = np.empty((n, dimension))
    y = np.empty(n, dtype=int)
    for i in range(n):
        X[i], y[i] = generator.sample()
    return X, y
