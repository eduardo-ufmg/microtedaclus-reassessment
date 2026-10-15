"""Execution time of the models with their selected hyperparameters, one task at a time.

The experiment runs its tasks in parallel, so the times it records depend on
the load of the machine. Here the tasks of the first runs are evaluated again,
each alone, with the hyperparameters that the experiment selected for them. A
model of the task first processes the first batches of the stream, so that what
it loads or creates on first use is in place, and a new model of the task is
then timed over the evaluation segment once.

Usage (from experimentos/modern_methodology), on a machine that does nothing else:
    python -m benchmark.timing [--runs FIRST LAST] [--models M ...]
Writes results_timing/batches/DATASET/MODEL/RUN.csv, with the columns of the
experiment, and continues from the tasks that are missing.
"""

import os

for _variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMBA_NUM_THREADS",
):
    os.environ[_variable] = "1"  # as in the experiment

import argparse
import json
import time
import warnings

import pandas as pd

from benchmark import analysis
from benchmark.datasets import load_stream
from benchmark.evaluation import prequential, warm_up
from benchmark.models import MODELS, build
from benchmark.runner import COST_RUNS, batches_file, stream_length, task_seeds
from benchmark.tuning import split

OUT = analysis.ROOT / "results_timing"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs", type=int, nargs=2, default=[0, COST_RUNS - 1])
    parser.add_argument(
        "--models", nargs="+", default=list(MODELS), choices=list(MODELS)
    )
    arguments = parser.parse_args()
    warnings.simplefilter("ignore")

    factor = analysis.length_factor(analysis.MAIN)
    tasks = pd.read_csv(analysis.MAIN / "tasks.csv")
    tasks = tasks[
        (tasks.status == "OK")
        & tasks.model.isin(arguments.models)
        & tasks.run.between(*arguments.runs)
    ].sort_values(["run", "dataset", "model"])
    missing = [
        task
        for task in tasks.itertuples()
        if not batches_file(OUT, task.dataset, task.model, task.run).exists()
    ]
    print(
        f"{len(missing)} tasks to time, {len(tasks) - len(missing)} already finished",
        flush=True,
    )
    started = time.perf_counter()
    for done, task in enumerate(missing, start=1):
        X, y = load_stream(task.dataset, task.run, stream_length(task.dataset, factor))
        _, (X_evaluation, y_evaluation) = split(X, y)
        _, model_seed = task_seeds(task.dataset, task.model, task.run)
        hyperparameters = json.loads(task.hyperparameters)

        def make_model():
            return build(MODELS[task.model], hyperparameters, model_seed)

        warm_up(make_model, X_evaluation)
        results = prequential(make_model(), X_evaluation, y_evaluation, measure=True)
        target = batches_file(OUT, task.dataset, task.model, task.run)
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(".partial")
        results.to_csv(partial, index=False)
        partial.replace(target)
        if done % 10 == 0 or done == len(missing):
            hours = (time.perf_counter() - started) / 3600
            remaining = hours / done * (len(missing) - done)
            print(
                f"{done}/{len(missing)} tasks, {hours:.2f} h elapsed, about {remaining:.1f} h left",
                flush=True,
            )


if __name__ == "__main__":
    main()
