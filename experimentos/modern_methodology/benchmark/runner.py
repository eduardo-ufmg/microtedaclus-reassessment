"""The experiment: every model tuned and evaluated on every stream.

A task is a (dataset, model, run). The hyperparameters of the model are
selected on the first part of the stream, and a new model with them is
evaluated prequentially on the rest. Tasks run in parallel processes, in the
order of their run, so the experiment can stop after any run with the same runs
complete for every dataset and model. A finished task is written at once, and a
new call continues from the tasks that are missing.

Usage (from experimentos/modern_methodology):
    python -m benchmark.runner OUT [--runs FIRST LAST] [--models M ...] [--datasets D ...]
                               [--workers N] [--trials N] [--length-factor F]
                               [--protocol rules|previous]
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
import csv
import json
import multiprocessing
import time
import warnings
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from benchmark.datasets import BASE_SEED, DATASETS, load_stream
from benchmark.evaluation import prequential
from benchmark.models import MODELS, build
from benchmark.tuning import PROTOCOLS, TRIALS, split

N_RUNS = 94
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
    length = DATASETS[dataset].length
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
):
    """Tune and evaluate one task; its row of tasks.csv."""
    warnings.simplefilter("ignore")
    row = {"dataset": dataset, "model": model, "run": run}
    try:
        X, y = load_stream(dataset, run, stream_length(dataset, length_factor))
        (X_tuning, y_tuning), (X_evaluation, y_evaluation) = split(X, y)
        search_seed, model_seed = task_seeds(dataset, model, run)
        hyperparameters, summary = PROTOCOLS[protocol](
            MODELS[model], X_tuning, y_tuning, search_seed, model_seed, trials
        )
        row |= summary
        if hyperparameters is None:
            return row | {"status": "ERROR: every trial failed"}
        row["hyperparameters"] = json.dumps(hyperparameters)

        started = time.perf_counter()
        results = prequential(
            build(MODELS[model], hyperparameters, model_seed),
            X_evaluation,
            y_evaluation,
            measure=True,
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


def finished_tasks(tasks_file: Path) -> set:
    if not tasks_file.exists():
        return set()
    with tasks_file.open(newline="") as stored:
        return {
            (r["dataset"], r["model"], int(r["run"])) for r in csv.DictReader(stored)
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("out", type=Path)
    parser.add_argument("--runs", type=int, nargs=2, default=[0, N_RUNS - 1])
    parser.add_argument(
        "--models", nargs="+", default=list(MODELS), choices=list(MODELS)
    )
    parser.add_argument(
        "--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS)
    )
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--trials", type=int, default=TRIALS)
    parser.add_argument("--length-factor", type=int, default=1)
    parser.add_argument("--protocol", default="rules", choices=list(PROTOCOLS))
    arguments = parser.parse_args()

    configuration = {
        "protocol": arguments.protocol,
        "trials": arguments.trials,
        "length_factor": arguments.length_factor,
    }
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

    is_new = not tasks_file.exists()
    started = time.perf_counter()
    with (
        tasks_file.open("a", newline="") as stored,
        ProcessPoolExecutor(
            max_workers=arguments.workers,
            mp_context=multiprocessing.get_context("spawn"),
            max_tasks_per_child=200,
        ) as pool,
    ):
        writer = csv.DictWriter(stored, fieldnames=TASK_COLUMNS)
        if is_new:
            writer.writeheader()
        futures = [
            pool.submit(
                run_task,
                str(arguments.out),
                *task,
                arguments.protocol,
                arguments.trials,
                arguments.length_factor,
            )
            for task in tasks
        ]
        for done, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            writer.writerow(row)
            stored.flush()
            os.fsync(stored.fileno())
            if row["status"] != "OK":
                print(
                    f"{row['dataset']} {row['model']} {row['run']}: {row['status']}",
                    flush=True,
                )
            if done % 50 == 0 or done == len(tasks):
                hours = (time.perf_counter() - started) / 3600
                remaining = hours / done * (len(tasks) - done)
                print(
                    f"{done}/{len(tasks)} tasks, {hours:.2f} h elapsed, about {remaining:.1f} h left",
                    flush=True,
                )


if __name__ == "__main__":
    main()
