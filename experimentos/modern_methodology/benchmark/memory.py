"""Memory of the models with their selected hyperparameters.

The tasks of the first runs of the experiment are evaluated again while every
allocation of their process is tracked, whether Python or compiled code makes
it. Two quantities are kept: the memory still allocated when the stream ends,
which is what the model holds, and the peak during the evaluation, which adds
the working memory of the steps of the model.

Every task runs in a new process, so that no task inherits what an earlier one
left allocated. The process first tracks a small evaluation, which takes the
allocations of the first tracking of a process; then a model of the task
processes the first batches of the stream untracked, so that what it loads or
creates on first use is in place; then a new model of the task is evaluated
once under tracking. Each process runs one thread, with the hash seed of Python
fixed.

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
from benchmark.evaluation import process, warm_up
from benchmark.models import MODELS, build
from benchmark.parallel import STALL_LIMIT, finished_tasks, run_tasks
from benchmark.runner import COST_RUNS, stream_length, task_seeds
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
_first_tracking_done = False


def tracked(make_model, X) -> tuple[int, int]:
    """Bytes that a new model holds after a stream, and the peak while it processes the stream."""
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

        def make_model():
            return build(MODELS[model], selected, model_seed)

        warm_up(make_model, X_evaluation)
        gc.collect()
        retained, peak = tracked(make_model, X_evaluation)
        return row | {
            "status": "OK",
            "memory_retained_bytes": retained,
            "memory_peak_bytes": peak,
        }
    except Exception as error:
        return row | {"status": f"ERROR: {error!r}"}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs", type=int, nargs=2, default=[0, COST_RUNS - 1])
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
        tasks_per_process=1,
    )


if __name__ == "__main__":
    main()
