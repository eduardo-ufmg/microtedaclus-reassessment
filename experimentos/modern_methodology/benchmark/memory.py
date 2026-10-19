"""Memory of the models with their selected hyperparameters.

The tasks of the first runs of the experiment are evaluated again while every
allocation of their process is tracked, whether Python or compiled code makes
it. Two quantities are kept: the memory still allocated when the stream ends,
which is what the model holds, and the peak during the evaluation, which adds
the working memory of the steps of the model.

Every task runs in a new process, so that no task inherits what an earlier one
left allocated. The process first tracks a small evaluation, which takes the
allocations of the first tracking of a process. A model of the task then
processes the whole evaluation segment untracked, because some models allocate
on first use well into the stream, and new models of the task are evaluated
three times under tracking, of which the smallest values are kept: what the
libraries keep allocated between uses adds to a measurement and never
subtracts from it. Each process runs one thread, with the hash seed of Python
fixed.

Usage (from experimentos/modern_methodology):
    python -m benchmark.memory [--results DIR] [--out DIR] [--runs FIRST LAST]
                               [--models M ...] [--workers N] [--stall-limit SECONDS]
                               [--memory-limit GIGABYTES]
The tasks and their hyperparameters come from the experiment in DIR, by default
results_main. Writes OUT/tasks.csv, by default under results_memory, with one
row per task, and continues from the tasks that are missing.
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
from benchmark.evaluation import process, warm_up
from benchmark.models import MODELS, build
from benchmark.parallel import STALL_LIMIT, finished_tasks, run_tasks
from benchmark.preprocessing import segments
from benchmark.runner import COST_RUNS, stream_length, task_seeds

OUT = analysis.ROOT / "results_memory"
TASK_COLUMNS = [
    "dataset",
    "model",
    "run",
    "status",
    "memory_retained_bytes",
    "memory_peak_bytes",
]
TRACKED_EVALUATIONS = 3
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
        _, (X_evaluation, _) = segments(dataset, run, stream_length(dataset, factor))
        _, model_seed = task_seeds(dataset, model, run)
        selected = json.loads(hyperparameters)

        def make_model():
            return build(MODELS[model], selected, model_seed)

        warm_up(make_model, X_evaluation, batches=None)
        gc.collect()
        evaluations = [
            tracked(make_model, X_evaluation) for _ in range(TRACKED_EVALUATIONS)
        ]
        retained = min(held for held, _ in evaluations)
        peak = min(highest for _, highest in evaluations)
        return row | {
            "status": "OK",
            "memory_retained_bytes": retained,
            "memory_peak_bytes": peak,
        }
    except Exception as error:
        return row | {"status": f"ERROR: {error!r}"}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results", type=Path, default=analysis.MAIN)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--runs", type=int, nargs=2, default=[0, COST_RUNS - 1])
    parser.add_argument(
        "--models", nargs="+", default=list(MODELS), choices=list(MODELS)
    )
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--stall-limit", type=int, default=STALL_LIMIT)
    parser.add_argument(
        "--memory-limit",
        type=float,
        help="gigabytes of memory per worker; a task that needs more fails",
    )
    arguments = parser.parse_args()

    factor = analysis.length_factor(arguments.results)
    selected = pd.read_csv(arguments.results / "tasks.csv")
    selected = selected[
        (selected.status == "OK")
        & selected.model.isin(arguments.models)
        & selected.run.between(*arguments.runs)
    ].sort_values(["run", "dataset", "model"])
    arguments.out.mkdir(exist_ok=True)
    tasks_file = arguments.out / "tasks.csv"
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
        memory_limit=(
            None
            if arguments.memory_limit is None
            else int(arguments.memory_limit * 2**30)
        ),
    )


if __name__ == "__main__":
    main()
