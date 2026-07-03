import optuna
from river.cluster import STREAMKMeans as CoreStreamKM


class StreamKM(CoreStreamKM):
    def __init__(
        self,
        chunk_size: int = 100,
        n_clusters: int = 8,
        halflife: float = 0.5,
        sigma: float = 3.0,
    ):
        super().__init__(
            chunk_size=chunk_size,
            n_clusters=n_clusters,
            halflife=halflife,
            sigma=sigma,
        )

    def _to_dict(self, x) -> dict:
        """Converts a 1D array-like structure to a dictionary for river."""
        return {f"x{i}": float(val) for i, val in enumerate(x)}

    def fit(self, X, y=None):
        for x in X:
            self.learn_one(self._to_dict(x))
        return self

    def predict(self, X):
        return (
            [self.predict_one(self._to_dict(x)) for x in X]
            if self.centers
            else [-1] * len(X)
        )

    def microcluster_count(self) -> int:
        return len(self.centers)


class StreamKMParameterSearchSpace:
    def sample_parameters(self, trial: optuna.Trial) -> dict:
        """Yields parameter suggestions utilizing Optuna's trial distributions."""
        return {
            "chunk_size": trial.suggest_int("chunk_size", 10, 1000),
            "n_clusters": trial.suggest_int("n_clusters", 2, 20),
            "halflife": trial.suggest_float("halflife", 0.01, 10),
            "sigma": trial.suggest_float("sigma", 0.001, 100, log=True),
        }
