"""MicroTEDAClus+ in the main experiment: the model of the shared package and its search space."""

import optuna

from pfc_shared.microtedaclus_plus import MicroTEDAClusPlus

__all__ = ["MicroTEDAClusPlus", "MicroTEDAClusPlusParameterSearchSpace"]


class MicroTEDAClusPlusParameterSearchSpace:
    def sample_parameters(self, trial: optuna.Trial) -> dict:
        return {
            "variance_limit": trial.suggest_float(
                "variance_limit", 1e-6, 1.0, log=True
            ),
            "decay": trial.suggest_int("decay", 10, 10000, log=True),
            "noise_distance": trial.suggest_float("noise_distance", 1.0, 10.0),
        }
