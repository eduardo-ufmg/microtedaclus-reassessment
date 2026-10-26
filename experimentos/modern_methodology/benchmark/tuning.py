"""Hyperparameter selection of a model on the first part of a stream.

Three protocols select the hyperparameters: that of the experiments, by the rules
`real2`; the same by the first version of the rules, `real`, which the pilot of the
real-data experiment used for some models; and that of the previous experiment,
kept to measure what the change of protocol does to the one model that both
experiments share.

By rules, the search sees the tuning segment only, which is what a deployment has before
the stream continues: its samples and their labels. The ranges come from the
segment by the rules of `benchmark.hyperparameters`. The objective is the
quantity that the evaluation reports, the batch ARI of a prequential pass, as
its median over the second half of the segment, so a configuration is judged
after it had the first half to form its summary. The objective has no term of
execution time and the search has no time limit: the selection depends on the
seed of the task and on nothing of the machine.
"""

import time

import numpy as np
import optuna

from benchmark.evaluation import prequential
from benchmark.hyperparameters import RULES, search_space, summary_limit
from benchmark.models import MicroTEDAClusPlus, build
from pfc_shared import previous_protocol
from pfc_shared.microtedaclus_plus import MicroTEDAClusPlus as PreviousMicroTEDAClusPlus

SPLIT_FRACTION = 0.2
TUNING_LIMIT = 10_000  # samples; the longest tuning segment of the synthetic experiment
TRIALS = 100
FAILED = -1.0  # objective of a configuration with which the model raises an error

optuna.logging.set_verbosity(optuna.logging.WARNING)


def split(X, y):
    """The tuning segment and the evaluation segment of a stream.

    The tuning segment is the first fifth of the stream, up to the limit of
    samples beyond which a search is not affordable for the slowest models.
    """
    at = min(int(SPLIT_FRACTION * len(X)), TUNING_LIMIT)
    return (X[:at], y[:at]), (X[at:], y[at:])


def objective_of(results) -> float:
    ari = results["ari"].to_numpy()
    return float(np.median(ari[len(ari) // 2 :]))


def tune(
    model_class,
    X,
    y,
    search_seed: int,
    model_seed: int,
    trials: int = TRIALS,
    rules: str = "real2",
):
    """Selection by rules: the hyperparameters and a summary of the search.

    The hyperparameters are None if every trial failed. They are kept as the
    model takes them, which differs from what the search samples for a range on
    the logit scale. A trial whose summary exceeds the limit of
    `benchmark.hyperparameters.summary_limit` fails, and the summary of the search
    carries the limit for the evaluation.
    """
    space = search_space(model_class, X, y, RULES[rules])
    limit = summary_limit(X, y)

    def objective(trial):
        hyperparameters = {
            name: of_name.suggest(trial, name) for name, of_name in space.items()
        }
        trial.set_user_attr("hyperparameters", hyperparameters)
        try:
            results = prequential(
                build(model_class, hyperparameters, model_seed),
                X,
                y,
                summary_limit=limit,
            )
        except Exception as error:
            trial.set_user_attr("error", repr(error))
            return FAILED
        return objective_of(results)

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=search_seed)
    )
    started = time.perf_counter()
    study.optimize(objective, n_trials=trials)
    failed = sum("error" in trial.user_attrs for trial in study.trials)
    summary = {
        "trials": len(study.trials),
        "failed_trials": failed,
        "objective": study.best_value,
        "tuning_seconds": time.perf_counter() - started,
        "summary_limit": limit,
    }
    selected = study.best_trial.user_attrs["hyperparameters"]
    return (selected if failed < len(study.trials) else None), summary


def tune_real(model_class, X, y, search_seed, model_seed, trials=TRIALS):
    """Selection by the first version of the rules."""
    return tune(model_class, X, y, search_seed, model_seed, trials, rules="real")


def tune_previous(
    model_class, X, y, search_seed: int, model_seed: int, trials: int = TRIALS
):
    """Selection of the previous experiment, which has a search space for MicroTEDAClus+ only."""
    if model_class is not MicroTEDAClusPlus:
        raise ValueError(f"the previous protocol has no search space for {model_class}")
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=search_seed)
    )
    started = time.perf_counter()
    study.optimize(
        lambda trial: previous_protocol.objective(
            trial,
            PreviousMicroTEDAClusPlus,
            previous_protocol.MicroTEDAClusPlusParameterSearchSpace,
            X,
            y,
        ),
        n_trials=trials,
        timeout=previous_protocol.OPT_TIMEOUT,
    )
    complete = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    summary = {
        "trials": len(study.trials),
        "failed_trials": len(study.trials) - len(complete),
        "objective": study.best_value if complete else float("nan"),
        "tuning_seconds": time.perf_counter() - started,
    }
    return (study.best_params if complete else None), summary


PROTOCOLS = {
    "real2": tune,
    "real": tune_real,
    "previous": tune_previous,
}
