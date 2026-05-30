import optuna
from river.cluster import DenStream as CoreDenStream


class DenStream(CoreDenStream):
    def __init__(
        self,
        decaying_factor: float = 0.25,
        beta: float = 0.75,
        mu: float = 2.0,
        epsilon: float = 0.02,
        n_samples_init: int = 1000,
        stream_speed: int = 100,
    ):
        super().__init__(
            decaying_factor=decaying_factor,
            beta=beta,
            mu=mu,
            epsilon=epsilon,
            n_samples_init=n_samples_init,
            stream_speed=stream_speed,
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
        return len(self.p_micro_clusters) + len(self.o_micro_clusters)


class DenStreamParameterSearchSpace:
    def sample_parameters(self, trial: optuna.Trial) -> dict:
        """Yields parameter suggestions utilizing Optuna's trial distributions."""
        return {
            "epsilon": trial.suggest_float("epsilon", 0.001, 0.01),
            "mu": trial.suggest_float("mu", 1.0, 20.0),
            "beta": trial.suggest_float("beta", 0.2, 0.4),
        }
