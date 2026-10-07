"""Reachability probe: what a search space contains versus what the tuning protocol selects.

The probe draws configurations at random from a search space and scores each of
them twice on every probed stream: with the objective of the hyperparameter
search, on the first 20% of the stream, and with the evaluation metrics, on the
remaining 80%. Comparing the two shows how much quality the search space holds
and how much of it the tuning objective is able to recognize. The same
configurations are scored on several runs, so that a configuration that is good
on one stream by chance can be told from one that is good on the dataset.

Usage (from experimentos/proposed_methodology):
    python -m model_improvement.probe SPACE DATASET [N_RUNS] [N_CONFIGURATIONS]
SPACE is `final` (the hyperparameters of the final model), `engine` (the final
model's hyperparameters plus design options explored in the chapter) or
`shrink:<exponent>` (the final model with a variance limit that shrinks as
(n / 100)^-exponent). The streams are the first N_RUNS development runs.
"""

import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
import optuna
import pandas as pd

import protocol
from model_improvement.engine import MicroTEDAClusEngine
from model_improvement.experiments import DEVELOPMENT_RUNS, FINAL
from model_improvement.harness import RESULTS_DIR, _single_threaded
from model_improvement.variants import VARIANTS, resolve
from streams import load_stream

PROBE_DIR = RESULTS_DIR / "_probe"
SEED = 123


class EngineSpace:
    """Hyperparameters of the final model plus design options explored in the chapter."""

    def sample_parameters(self, trial):
        return {
            "variance_limit": trial.suggest_float(
                "variance_limit", 1e-6, 1.0, log=True
            ),
            "decay": trial.suggest_int("decay", 10, 10000, log=True),
            "noise_m": trial.suggest_float("noise_m", 1.0, 10.0),
            "single_assign": trial.suggest_categorical("single_assign", [True, False]),
            "cap_variance": trial.suggest_categorical("cap_variance", [True, False]),
            "density": trial.suggest_categorical("density", ["last", "peak"]),
            "life_mode": trial.suggest_categorical(
                "life_mode", ["inactive", "recency", "recency_inactive"]
            ),
            "activation_factor": trial.suggest_categorical(
                "activation_factor", [0.0, 0.25, 0.5, 0.75, 1.0]
            ),
            "link": trial.suggest_categorical("link", [1.0, 1.5, 2.0]),
        }


def search_space(name: str):
    """Return (model_class, search_space_class) of a probe space."""
    final_class, final_space = resolve(FINAL)
    if name == "final":
        return final_class, final_space
    if name.startswith("shrink:"):
        return partial(final_class, shrink_exp=float(name.split(":")[1])), final_space
    if name == "engine":
        fixed = {
            k: v for k, v in VARIANTS[FINAL].options.items() if k != "single_assign"
        }
        return partial(MicroTEDAClusEngine, **fixed), EngineSpace
    raise ValueError(name)


def score(task):
    space_name, dataset, run, configuration, params = task
    model_class, search_space_class = search_space(space_name)
    X, y = load_stream(dataset, run)
    split = int(protocol.SPLIT_FRACTION * len(X))
    try:
        tuning = protocol.objective(
            optuna.trial.FixedTrial(params),
            model_class,
            search_space_class,
            X[:split],
            y[:split],
        )
        results = pd.DataFrame(
            protocol.evaluate(model_class(**params), X[split:], y[split:])
        )
        ari = results["ari"].to_numpy()
        return {
            "configuration": configuration,
            "run": run,
            **params,
            "tuning_objective": tuning,
            "ari_overall": float(np.median(ari)),
            "ari_steadystate": float(ari[-2]),
            "microclusters": int(results["microcluster_count"].iloc[-2]),
        }
    except Exception:
        return None


def summarize(df: pd.DataFrame) -> str:
    """Contrast the configurations the tuning objective selects with the best of the space."""
    selected = df.loc[df.groupby("run")["tuning_objective"].idxmax()]
    best_per_run = df.groupby("run")["ari_overall"].max()
    complete = df.groupby("configuration").filter(
        lambda rows: len(rows) == df["run"].nunique()
    )
    across_runs = complete.groupby("configuration")["ari_overall"].median()
    correlation = df.groupby("run")[["tuning_objective", "ari_overall"]].apply(
        lambda rows: rows.corr(method="spearman").iloc[0, 1]
    )
    return (
        f"{df['configuration'].nunique()} configurations on {df['run'].nunique()} runs | "
        f"Overall ARI of the configuration selected in each run, median {selected.ari_overall.median():.3f} | "
        f"best Overall ARI of each run, median {best_per_run.median():.3f} | "
        f"best median Overall ARI of one configuration across the runs {across_runs.max():.3f} | "
        f"Spearman, median {correlation.median():.2f}"
    )


def main(space_name: str, dataset: str, n_runs: int = 4, n: int = 400):
    _, search_space_class = search_space(space_name)
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=SEED))
    configurations = [
        search_space_class().sample_parameters(study.ask()) for _ in range(n)
    ]
    tasks = [
        (space_name, dataset, run, i, c)
        for run in list(DEVELOPMENT_RUNS)[:n_runs]
        for i, c in enumerate(configurations)
    ]
    with ProcessPoolExecutor(max_workers=10, initializer=_single_threaded) as pool:
        rows = [r for r in pool.map(score, tasks, chunksize=4) if r]
    df = pd.DataFrame(rows)
    PROBE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROBE_DIR / f"{space_name.replace(':', '_')}_{dataset}.csv", index=False)
    print(f"{space_name} on {dataset}: {summarize(df)}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], *map(int, sys.argv[3:5]))
