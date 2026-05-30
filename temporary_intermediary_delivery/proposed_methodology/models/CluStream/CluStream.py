import optuna
from river.cluster import CluStream as CoreCluStream


class CluStream(CoreCluStream):
    def __init__(
        self,
        n_macro_clusters: int = 5,
        max_micro_clusters: int = 100,
        micro_cluster_r_factor: int = 2,
        time_window: int = 1000,
        time_gap: int = 100,
        seed: int | None = None,
        **kwargs,
    ):
        super().__init__(
            n_macro_clusters=n_macro_clusters,
            max_micro_clusters=max_micro_clusters,
            micro_cluster_r_factor=micro_cluster_r_factor,
            time_window=time_window,
            time_gap=time_gap,
            seed=seed,
            **kwargs,
        )

    def _to_dict(self, x) -> dict:
        """Converts a 1D array-like structure to a dictionary for river."""
        return {f"x{i}": float(val) for i, val in enumerate(x)}

    def fit(self, X, y=None):
        for x in X:
            self.learn_one(self._to_dict(x))
        return self

    def predict(self, X):
        return [self.predict_one(self._to_dict(x)) for x in X]

    def microcluster_count(self) -> int:
        return len(self.micro_clusters)


class CluStreamParameterSearchSpace:
    def sample_parameters(self, trial: optuna.Trial) -> dict:
        """Yields parameter suggestions utilizing Optuna's trial distributions."""
        return {
            "max_micro_clusters": trial.suggest_int("max_micro_clusters", 10, 1000),
            "time_window": trial.suggest_int("time_window", 100, 1000),
            "micro_cluster_r_factor": trial.suggest_int(
                "micro_cluster_r_factor", 1, 10
            ),
        }
