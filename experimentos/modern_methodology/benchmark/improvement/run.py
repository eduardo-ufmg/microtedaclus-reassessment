"""Tune and evaluate a candidate on the development runs of the six real datasets.

The task of a candidate is that of the real-data experiment: standardized
segments, the widened rules, the summary limit, 100 trials. Every candidate of a
dataset and run uses the same seeds, those of MicroTEDAClus+ in that task, so
that a candidate and its control differ in the model only.

Usage (from experimentos/modern_methodology):
    python -m benchmark.improvement.run CANDIDATE [--runs FIRST LAST] [--datasets D ...]
                                        [--workers N] [--trials N] [--memory-limit GB]
                                        [--stall-limit SECONDS]
Writes results_improvement/CANDIDATE/tasks.csv and batches/DATASET/CANDIDATE/RUN.csv,
and continues from the tasks already written.
"""

import os

for _variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMBA_NUM_THREADS",
):
    os.environ[_variable] = "1"

import argparse
import json
import time
import warnings
from pathlib import Path

from benchmark.analysis import ROOT
from benchmark.evaluation import prequential
from benchmark.improvement.candidates import CANDIDATES, model_class
from benchmark.parallel import finished_tasks, run_tasks
from benchmark.preprocessing import segments
from benchmark.runner import (
    TASK_COLUMNS,
    batches_file,
    configuration_matches,
    task_seeds,
)
from benchmark.tuning import PROTOCOLS, TRIALS

OUT = ROOT / "results_improvement"
DATASETS = [
    "insects_abrupt",
    "insects_incremental",
    "insects_incremental_reoccurring",
    "gas_drift",
    "pendigits",
    "shuttle",
]
DEVELOPMENT_RUNS = (32, 63)
SEED_MODEL = "MicroTEDAClusPlus"  # the seeds of every candidate


def run_task(out: str, candidate: str, dataset: str, run: int, trials: int):
    warnings.simplefilter("ignore")
    row = {"dataset": dataset, "model": candidate, "run": run}
    try:
        (X_tuning, y_tuning), (X_evaluation, y_evaluation) = segments(dataset, run)
        search_seed, model_seed = task_seeds(dataset, SEED_MODEL, run)
        model = model_class(candidate)
        hyperparameters, summary = PROTOCOLS["widened"](
            model, X_tuning, y_tuning, search_seed, model_seed, trials
        )
        limit = summary.pop("summary_limit", None)
        row |= summary
        if hyperparameters is None:
            return row | {"status": "ERROR: every trial failed"}
        row["hyperparameters"] = json.dumps(hyperparameters)

        started = time.perf_counter()
        results = prequential(
            model(**hyperparameters),
            X_evaluation,
            y_evaluation,
            measure=True,
            summary_limit=limit,
        )
        row["evaluation_seconds"] = time.perf_counter() - started

        target = batches_file(Path(out), dataset, candidate, run)
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(".partial")
        results.to_csv(partial, index=False)
        partial.replace(target)
        return row | {"status": "OK"}
    except Exception as error:
        return row | {"status": f"ERROR: {error!r}"}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("candidate", choices=list(CANDIDATES))
    parser.add_argument("--runs", type=int, nargs=2, default=list(DEVELOPMENT_RUNS))
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--trials", type=int, default=TRIALS)
    parser.add_argument("--memory-limit", type=float, default=1.5)
    parser.add_argument("--stall-limit", type=int, default=3600)
    parser.add_argument("--out", type=Path, default=OUT)
    arguments = parser.parse_args()

    candidate = CANDIDATES[arguments.candidate]
    out = arguments.out / candidate.name
    configuration = {
        "protocol": "widened",
        "trials": arguments.trials,
        "length_factor": 1,
        "candidate": candidate.name,
        "engine": candidate.engine,
        "options": candidate.options,
    }
    if not configuration_matches(out, configuration):
        parser.error(f"{out} holds results of another configuration")

    tasks_file = out / "tasks.csv"
    finished = finished_tasks(tasks_file)
    tasks = [
        (str(out), candidate.name, dataset, run, arguments.trials)
        for run in range(arguments.runs[0], arguments.runs[1] + 1)
        for dataset in arguments.datasets
        if (dataset, candidate.name, run) not in finished
    ]
    print(
        f"{candidate.name}: {len(tasks)} tasks to run, {len(finished)} already finished",
        flush=True,
    )
    run_tasks(
        run_task,
        tasks,
        tasks_file,
        TASK_COLUMNS,
        arguments.workers,
        arguments.stall_limit,
        memory_limit=int(arguments.memory_limit * 2**30),
    )
    print("The candidate is complete.", flush=True)


if __name__ == "__main__":
    main()
