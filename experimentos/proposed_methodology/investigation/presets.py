"""Named models that came out of the investigation, in the project's model/search-space convention.

    MicroTEDAClusCorrectedNext  stage 2: the reference behaviour plus three corrections
    MicroTEDAClusImprovedNext   stage 3: the corrected model plus five improvements

Both are thin configurations of investigation.engine.TedaClus. The stage 1 model
(behaviour-preserving) is models.MicroTEDAClusVectorized.MicroTEDAClusVectorized.
"""

import optuna

from investigation.engine import TedaClus

CORRECTIONS = {
    "membership": "nearest",  # count-free eccentricity d^2/sigma^2 instead of density-weighted 1 - zeta
    "density": "peak",  # intrinsic density 1/zeta(mu) = 2S instead of the last absorbed sample's
    "prune_skip": False,  # pruning no longer skips the microcluster after a removed one
}

IMPROVEMENTS = {
    "relative_limit": True,  # variance_limit is a fraction of the stream's running global variance
    "dim_penalty": False,  # no 2/d scaling of distances
    "cap_variance": True,  # variance_limit bounds microclusters of every size, not only pairs
    "stable_labels": True,  # macroclusters keep their identity across updates
    # noise_m (tuned): a sample farther than noise_m sigma from every active microcluster is labelled -1
}


class MicroTEDAClusCorrectedNext(TedaClus):
    def __init__(self, variance_limit=0.001, decay=100):
        super().__init__(variance_limit, decay, **CORRECTIONS)


class MicroTEDAClusCorrectedNextParameterSearchSpace:
    def sample_parameters(self, trial: optuna.Trial) -> dict:
        return {
            "variance_limit": trial.suggest_float(
                "variance_limit", 0.0001, 100.0, log=True
            ),
            "decay": trial.suggest_int("decay", 10, 10000, log=True),
        }


class MicroTEDAClusImprovedNext(TedaClus):
    def __init__(self, variance_limit=0.01, decay=100, noise_m=6.0):
        super().__init__(
            variance_limit, decay, noise_m=noise_m, **CORRECTIONS, **IMPROVEMENTS
        )


class MicroTEDAClusImprovedNextParameterSearchSpace:
    def sample_parameters(self, trial: optuna.Trial) -> dict:
        return {
            "variance_limit": trial.suggest_float(
                "variance_limit", 1e-6, 1.0, log=True
            ),
            "decay": trial.suggest_int("decay", 10, 10000, log=True),
            "noise_m": trial.suggest_float("noise_m", 1.0, 10.0),
        }
