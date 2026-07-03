"""
CEDAS: Clustering of Evolving Data-streams into Arbitrary Shapes.

Ported from:
    Hyde, R., Angelov, P., MacKenzie, A.R. (2017).
    "Fully online clustering of evolving data streams into arbitrarily
    shaped clusters." Information Sciences 382-383, 96-114.
    (Algorithms 1-4, Section 3).
"""

import math
from collections.abc import Sequence

import optuna


class CEDAS:
    def __init__(self, r0: float, decay_period: int, min_samples: int):
        if r0 <= 0:
            raise ValueError("r0 must be > 0")
        if decay_period <= 0:
            raise ValueError("decay_period must be > 0")
        if min_samples < 1:
            raise ValueError("min_samples must be >= 1")

        self.r0 = float(r0)
        self.decay_rate = 1.0 / float(decay_period)
        self.min_samples = int(min_samples)

        self._next_id = 0
        self.centers: dict[int, list[float]] = {}
        self.counts: dict[int, int] = {}
        self.energy: dict[int, float] = {}
        self.macro: dict[int, int] = {}
        self.edges: dict[int, set[int]] = {}

        self._next_macro = 0
        self._n_features: int | None = None
        self._keys = None

    def fit(self, X, y=None):
        for x in X:
            self.learn_one(self._to_dict(x))
        return self

    def predict(self, X):
        return [self.predict_one(self._to_dict(x)) for x in X]

    def microcluster_count(self) -> int:
        return len(self.p_micro_clusters) + len(self.o_micro_clusters)

    @property
    def p_micro_clusters(self) -> list[int]:
        return [i for i, c in self.counts.items() if c >= self.min_samples]

    @property
    def o_micro_clusters(self) -> list[int]:
        return [i for i, c in self.counts.items() if c < self.min_samples]

    @staticmethod
    def _to_dict(x: Sequence[float] | dict[str, float]) -> dict[str, float]:
        if isinstance(x, dict):
            return x
        return {f"x{i}": float(v) for i, v in enumerate(x)}

    def _vec(self, x: dict[str, float]) -> list[float]:
        keys = sorted(x.keys())
        if self._n_features is None:
            self._n_features = len(keys)
            self._keys = keys
        elif keys != self._keys:
            raise ValueError(f"Feature mismatch: expected {self._keys}, got {keys}")
        return [x[k] for k in self._keys] if self._keys else []

    @staticmethod
    def _dist(a: list[float], b: list[float]) -> float:
        return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))

    def learn_one(self, x: dict[str, float]) -> "CEDAS":
        v = self._vec(x)
        if not self.centers:
            self._init_first(v)
            return self
        modified_id = self._update_microclusters(v)
        self._kill_microclusters()
        self._update_graph(modified_id)
        return self

    def predict_one(self, x: dict[str, float]) -> int:
        if not self.centers:
            return -1
        v = self._vec(x)
        best_id, best_d = None, math.inf
        for cid, c in self.centers.items():
            d = self._dist(v, c)
            if d < best_d:
                best_id, best_d = cid, d
        if best_id is None or best_d >= self.r0:
            return -1
        if self.counts[best_id] < self.min_samples:
            return -1
        return self.macro[best_id]

    def _new_id(self) -> int:
        cid = self._next_id
        self._next_id += 1
        return cid

    def _new_macro(self) -> int:
        m = self._next_macro
        self._next_macro += 1
        return m

    def _init_first(self, v: list[float]) -> None:
        cid = self._new_id()
        self.centers[cid] = list(v)
        self.counts[cid] = 1
        self.energy[cid] = 1.0
        self.macro[cid] = self._new_macro()
        self.edges[cid] = set()

    def _update_microclusters(self, v: list[float]) -> int:
        best_id, best_d = None, math.inf
        for cid, c in self.centers.items():
            d = self._dist(v, c)
            if d < best_d:
                best_id, best_d = cid, d

        if best_d < self.r0:

            if best_id is None:
                raise RuntimeError("best_id is None despite best_d < r0")

            self.energy[best_id] = 1.0
            self.counts[
                best_id
            ] += 1  # increment BEFORE centre update (matches CEDAS.m order)
            if (
                best_d > self.r0 / 2.0
            ):  # shell region -> recursive mean over TOTAL count
                n = self.counts[best_id]
                c = self.centers[best_id]
                self.centers[best_id] = [
                    ((n - 1) * ci + vi) / n for ci, vi in zip(c, v)
                ]
            # kernel region: count updated, centre unchanged
            return best_id
        else:
            cid = self._new_id()
            self.centers[cid] = list(v)
            self.counts[cid] = 1
            self.energy[cid] = 1.0
            self.macro[cid] = self._new_macro()
            self.edges[cid] = set()
            return cid

    def _kill_microclusters(self) -> None:
        dead = []
        for cid in list(self.energy.keys()):
            self.energy[cid] -= self.decay_rate
            if self.energy[cid] < 0:
                dead.append(cid)
        if not dead:
            return
        affected_neighbours: set[int] = set()
        dead_set = set(dead)
        for cid in dead:
            affected_neighbours |= self.edges.get(cid, set())
            for nb in self.edges.get(cid, set()):
                self.edges[nb].discard(cid)
            del self.centers[cid]
            del self.counts[cid]
            del self.energy[cid]
            del self.macro[cid]
            del self.edges[cid]
        affected_neighbours -= dead_set
        if affected_neighbours:
            self._relabel_components(affected_neighbours)

    def _update_graph(self, modified_id: int) -> None:
        if modified_id not in self.centers:
            return
        if self.counts[modified_id] < self.min_samples:
            return

        new_edges = False
        c = self.centers[modified_id]
        for cid, c2 in self.centers.items():
            if cid == modified_id or cid in self.edges[modified_id]:
                continue
            if self.counts[cid] < self.min_samples:
                continue
            if self._dist(c, c2) <= 1.5 * self.r0:
                self.edges[modified_id].add(cid)
                self.edges[cid].add(modified_id)
                new_edges = True

        if new_edges:
            self._relabel_components({modified_id})

    def _relabel_components(self, seeds: set[int]) -> None:
        visited: set[int] = set()
        for seed in seeds:
            if seed in visited or seed not in self.centers:
                continue
            component = self._bfs(seed)
            visited |= component
            label = self._new_macro()
            for cid in component:
                self.macro[cid] = label

    def _bfs(self, start: int) -> set[int]:
        seen = {start}
        stack = [start]
        while stack:
            cur = stack.pop()
            for nb in self.edges[cur]:
                if nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        return seen


class CEDASParameterSearchSpace:
    def sample_parameters(self, trial: optuna.Trial) -> dict:
        """Yields parameter suggestions utilizing Optuna's trial distributions."""
        return {
            "r0": trial.suggest_float("r0", 0.0001, 50.0, log=True),
            "decay_period": trial.suggest_int("decay_period", 10, 10000, log=True),
            "min_samples": trial.suggest_int("min_samples", 1, 400),
        }
