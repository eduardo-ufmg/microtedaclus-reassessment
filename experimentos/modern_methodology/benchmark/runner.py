"""The experiment: every model tuned and evaluated on every stream.

A task is a (dataset, model, run). The hyperparameters of the model are
selected on the first part of the stream, and a new model with them is
evaluated prequentially on the rest. Tasks run in parallel processes, in the
order of their run, so the experiment can stop after any run with the same runs
complete for every dataset and model. A finished task is written at once, and a
new call continues from the tasks that are missing. The run ends with an error
if no task finishes within the stall limit. With --tune-only, a task selects its
hyperparameters and stops, without the evaluation, as the pilot that checks the
search ranges does.

Usage (from experimentos/modern_methodology):
    python -m benchmark.runner OUT [--runs FIRST LAST] [--models M ...] [--datasets D ...]
                               [--workers N] [--trials N] [--length-factor F]
                               [--protocol rules|widened|real|real2|previous] [--stall-limit SECONDS]
                               [--memory-limit GIGABYTES] [--tune-only]
OUT receives tasks.csv, with one row per task, and batches/DATASET/MODEL/RUN.csv.
It holds the results of one configuration: a protocol, a number of trials and a
length factor.
"""

import os

for _variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMBA_NUM_THREADS",
):
    os.environ[_variable] = (
        "1"  # one thread per process: the processes are the parallelism
    )

import argparse
import json
import time
import warnings
import zlib
from pathlib import Path

import numpy as np

from benchmark.datasets import BASE_SEED, DATASETS, STREAMS
from benchmark.evaluation import prequential
from benchmark.models import MODELS, build
from benchmark.parallel import STALL_LIMIT, finished_tasks, run_tasks
from benchmark.preprocessing import segments
from benchmark.tuning import PROTOCOLS, TRIALS

N_RUNS = 94
COST_RUNS = 10  # first runs whose tasks the timing and memory passes measure
TASK_COLUMNS = [
    "dataset",
    "model",
    "run",
    "status",
    "trials",
    "failed_trials",
    "objective",
    "tuning_seconds",
    "evaluation_seconds",
    "hyperparameters",
]


def task_seeds(dataset: str, model: str, run: int) -> tuple[int, int]:
    """Seeds of the search and of the model of a task."""
    entropy = [BASE_SEED, zlib.crc32(dataset.encode()), zlib.crc32(model.encode()), run]
    search, of_model = np.random.SeedSequence(entropy).generate_state(2)
    return int(search), int(of_model)


def stream_length(dataset: str, length_factor: int) -> int | None:
    """Length of a generated stream; None for a file, which has its own."""
    length = STREAMS[dataset].length
    return None if length is None else length * length_factor


def batches_file(out: Path, dataset: str, model: str, run: int) -> Path:
    return out / "batches" / dataset / model / f"{run:03d}.csv"


def run_task(
    out: str,
    dataset: str,
    model: str,
    run: int,
    protocol: str,
    trials: int,
    length_factor: int,
    tune_only: bool = False,
):
    """Tune and evaluate one task, or only tune it; its row of tasks.csv."""
    warnings.simplefilter("ignore")
    row = {"dataset": dataset, "model": model, "run": run}
    try:
        (X_tuning, y_tuning), (X_evaluation, y_evaluation) = segments(
            dataset, run, stream_length(dataset, length_factor)
        )
        search_seed, model_seed = task_seeds(dataset, model, run)
        hyperparameters, summary = PROTOCOLS[protocol](
            MODELS[model], X_tuning, y_tuning, search_seed, model_seed, trials
        )
        limit = summary.pop("summary_limit", None)
        row |= summary
        if hyperparameters is None:
            return row | {"status": "ERROR: every trial failed"}
        row["hyperparameters"] = json.dumps(hyperparameters)
        if tune_only:
            return row | {"status": "OK"}

        started = time.perf_counter()
        results = prequential(
            build(MODELS[model], hyperparameters, model_seed),
            X_evaluation,
            y_evaluation,
            measure=True,
            summary_limit=limit,
        )
        row["evaluation_seconds"] = time.perf_counter() - started

        target = batches_file(Path(out), dataset, model, run)
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(".partial")
        results.to_csv(partial, index=False)
        partial.replace(target)
        return row | {"status": "OK"}
    except Exception as error:
        return row | {"status": f"ERROR: {error!r}"}


def configuration_matches(out: Path, configuration: dict) -> bool:
    """Whether OUT is new or was written with the same configuration, which it then records."""
    stored = out / "configuration.json"
    if stored.exists():
        return json.loads(stored.read_text()) == configuration
    out.mkdir(parents=True, exist_ok=True)
    stored.write_text(json.dumps(configuration, indent=2) + "\n")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("out", type=Path)
    parser.add_argument("--runs", type=int, nargs=2, default=[0, N_RUNS - 1])
    parser.add_argument(
        "--models", nargs="+", default=list(MODELS), choices=list(MODELS)
    )
    parser.add_argument(
        "--datasets", nargs="+", default=list(DATASETS), choices=list(STREAMS)
    )
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--trials", type=int, default=TRIALS)
    parser.add_argument("--length-factor", type=int, default=1)
    parser.add_argument("--protocol", default="rules", choices=list(PROTOCOLS))
    parser.add_argument("--stall-limit", type=int, default=STALL_LIMIT)
    parser.add_argument(
        "--memory-limit",
        type=float,
        default=None,
        help="gigabytes of memory per worker; a task that needs more fails",
    )
    parser.add_argument(
        "--tune-only",
        action="store_true",
        help="select the hyperparameters of every task without evaluating it",
    )
    arguments = parser.parse_args()

    configuration = {
        "protocol": arguments.protocol,
        "trials": arguments.trials,
        "length_factor": arguments.length_factor,
    }
    if arguments.tune_only:
        configuration["tune_only"] = True
    if not configuration_matches(arguments.out, configuration):
        parser.error(f"{arguments.out} holds results of another configuration")

    tasks_file = arguments.out / "tasks.csv"
    finished = finished_tasks(tasks_file)
    tasks = [
        (dataset, model, run)
        for run in range(arguments.runs[0], arguments.runs[1] + 1)
        for dataset in arguments.datasets
        for model in arguments.models
        if (dataset, model, run) not in finished
    ]
    print(f"{len(tasks)} tasks to run, {len(finished)} already finished", flush=True)

    run_tasks(
        run_task,
        [
            (
                str(arguments.out),
                *task,
                arguments.protocol,
                arguments.trials,
                arguments.length_factor,
                arguments.tune_only,
            )
            for task in tasks
        ],
        tasks_file,
        TASK_COLUMNS,
        arguments.workers,
        arguments.stall_limit,
        memory_limit=(
            None
            if arguments.memory_limit is None
            else int(arguments.memory_limit * 2**30)
        ),
    )


if __name__ == "__main__":
    main()
