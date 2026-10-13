"""Hyperparameter selection of the previous experiment.

The hyperparameters of a model are searched on the first part of the stream by
Bayesian optimization, for a number of trials or until a time limit, within
ranges chosen for each model. The objective is the ARI of a prequential pass
after a burn-in, less a penalty on its execution time.
"""

import time

import numpy as np
import optuna
from sklearn.metrics import adjusted_rand_score

BATCH_SIZE = 100
OPT_TRIALS = 100
OPT_TIMEOUT = 300
BURN_IN_FRACTION = 0.05
SPLIT_FRACTION = 0.2


def objective(trial, model_class, search_space_class, X_opt, y_opt):
    """Optuna objective: prequential ARI after the burn-in, penalized by execution time."""
    params = search_space_class().sample_parameters(trial)
    try:
        model = model_class(**params)
    except Exception:
        raise optuna.exceptions.TrialPruned()

    burn_in_idx = int(BURN_IN_FRACTION * len(X_opt))
    y_pred_all = []
    num_batches = len(X_opt) // BATCH_SIZE

    t0 = time.process_time_ns()
    try:
        for b in range(num_batches):
            start_idx, end_idx = b * BATCH_SIZE, (b + 1) * BATCH_SIZE
            X_batch = X_opt[start_idx:end_idx]

            preds = model.predict(X_batch)
            y_pred_all.extend(preds)
            model.fit(X_batch)
    except Exception:
        raise optuna.exceptions.TrialPruned()

    t_total_sec = (time.process_time_ns() - t0) / 1e9

    y_pred_eval = np.array(y_pred_all)[burn_in_idx:]
    y_opt_eval = y_opt[: len(y_pred_all)][burn_in_idx:]

    ari = adjusted_rand_score(y_opt_eval, y_pred_eval)
    return ari - (1e-3 * t_total_sec)


class MicroTEDAClusPlusParameterSearchSpace:
    def sample_parameters(self, trial: optuna.Trial) -> dict:
        return {
            "variance_limit": trial.suggest_float(
                "variance_limit", 1e-6, 1.0, log=True
            ),
            "decay": trial.suggest_int("decay", 10, 10000, log=True),
            "noise_distance": trial.suggest_float("noise_distance", 1.0, 10.0),
        }
