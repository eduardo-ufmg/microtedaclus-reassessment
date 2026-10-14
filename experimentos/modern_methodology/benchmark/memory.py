"""Memory of the models with their selected hyperparameters.

Every finished task of the experiment is evaluated again while every allocation
of its process is tracked, whether Python or compiled code makes it. Two
quantities are kept: the memory still allocated when the stream ends, which is
what the model holds, and the peak during the evaluation, which adds the
working memory of the steps of the model.

For the numbers to repeat, a task is evaluated once without tracking, so that
whatever is loaded or created on first use is in place, and then tracked three
times, of which the smallest values are kept: what the libraries keep allocated
between uses adds to a measurement and never subtracts from it. Each process
runs one thread, with the hash seed of Python fixed, and tracks a small
evaluation when it starts, which takes the allocations of the first tracking of
a process.

Usage (from experimentos/modern_methodology):
    python -m benchmark.memory [--runs FIRST LAST] [--models M ...] [--workers N]
                               [--stall-limit SECONDS]
Writes results_memory/tasks.csv, with one row per task, and continues from the
tasks that are missing.
"""

import os

for _variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMBA_NUM_THREADS",
):
    os.environ[_variable] = "1"  # the threads of a library have memory of their own
os.environ["PYTHONHASHSEED"] = "0"  # read by the worker processes when they start

import argparse
import gc
import json
import tempfile
import warnings
from pathlib import Path

import memray
import numpy as np
import pandas as pd

from benchmark import analysis
from benchmark.datasets import load_stream
from benchmark.evaluation import process
from benchmark.models import MODELS, build
from benchmark.parallel import STALL_LIMIT, finished_tasks, run_tasks
from benchmark.runner import N_RUNS, stream_length, task_seeds
from benchmark.tuning import split

OUT = analysis.ROOT / "results_memory"
TASK_COLUMNS = [
    "dataset",
    "model",
    "run",
    "status",
    "memory_retained_bytes",
    "memory_peak_bytes",
]
REPETITIONS = 3  # tracked evaluations of a task
_first_tracking_done = False


def _tracked_once(make_model, X) -> tuple[int, int]:
    with tempfile.TemporaryDirectory() as directory:
        trace = Path(directory) / "allocations.bin"
        with memray.Tracker(
            trace,
            trace_python_allocators=True,
            file_format=memray.FileFormat.AGGREGATED_ALLOCATIONS,
        ):
            model = make_model()
            process(model, X)
            gc.collect()
        reader = memray.FileReader(trace)
        retained = sum(
            record.size
            for record in reader.get_leaked_allocation_records(merge_threads=True)
        )
        peak = reader.metadata.peak_memory
        reader.close()
    del model
    return retained, peak


def tracked(make_model, X) -> tuple[int, int]:
    """Bytes that a model holds after a stream, and the peak while it processes the stream."""
    process(
        make_model(), X
    )  # everything loaded or created on first use is now in place
    gc.collect()
    passes = [_tracked_once(make_model, X) for _ in range(REPETITIONS)]
    return min(retained for retained, _ in passes), min(peak for _, peak in passes)


def _first_tracking():
    """Track a small evaluation once in a process, before its first measurement."""
    global _first_tracking_done
    if not _first_tracking_done:
        X = np.random.default_rng(0).normal(size=(300, 2))
        tracked(lambda: build(MODELS["MiniBatchKMeans"], {"n_clusters": 2}, 0), X)
        _first_tracking_done = True


def measure_task(dataset: str, model: str, run: int, hyperparameters: str, factor: int):
    """Measure one task; its row of tasks.csv."""
    warnings.simplefilter("ignore")
    row = {"dataset": dataset, "model": model, "run": run}
    try:
        _first_tracking()
        X, y = load_stream(dataset, run, stream_length(dataset, factor))
        _, (X_evaluation, _) = split(X, y)
        _, model_seed = task_seeds(dataset, model, run)
        selected = json.loads(hyperparameters)
        retained, peak = tracked(
            lambda: build(MODELS[model], selected, model_seed), X_evaluation
        )
        return row | {
            "status": "OK",
            "memory_retained_bytes": retained,
            "memory_peak_bytes": peak,
        }
    except Exception as error:
        return row | {"status": f"ERROR: {error!r}"}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs", type=int, nargs=2, default=[0, N_RUNS - 1])
    parser.add_argument(
        "--models", nargs="+", default=list(MODELS), choices=list(MODELS)
    )
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--stall-limit", type=int, default=STALL_LIMIT)
    arguments = parser.parse_args()

    factor = analysis.length_factor(analysis.MAIN)
    selected = pd.read_csv(analysis.MAIN / "tasks.csv")
    selected = selected[
        (selected.status == "OK")
        & selected.model.isin(arguments.models)
        & selected.run.between(*arguments.runs)
    ].sort_values(["run", "dataset", "model"])
    OUT.mkdir(exist_ok=True)
    tasks_file = OUT / "tasks.csv"
    finished = finished_tasks(tasks_file)
    tasks = [
        (task.dataset, task.model, task.run, task.hyperparameters, factor)
        for task in selected.itertuples()
        if (task.dataset, task.model, task.run) not in finished
    ]
    print(
        f"{len(tasks)} tasks to measure, {len(finished)} already finished", flush=True
    )
    run_tasks(
        measure_task,
        tasks,
        tasks_file,
        TASK_COLUMNS,
        arguments.workers,
        arguments.stall_limit,
    )


if __name__ == "__main__":
    main()
