"""What a model does on the real datasets: a replay of its selected configurations.

Each finished task of a model in an experiment is replayed with the
hyperparameters that its search selected, on the same evaluation segment, and at
every batch the prediction and the state of the model are recorded: the share
of the samples labeled as noise, the numbers of predicted clusters and of
classes, the homogeneity and completeness of the prediction against the classes
over the samples that it clusters, and, for the models with microclusters, the
numbers of microclusters, of mature ones (three samples or more) and of active
ones. A task keeps the median of each quantity over its batches.

Usage (from experimentos/modern_methodology):
    python -m benchmark.improvement.diagnose RESULTS MODEL [--runs FIRST LAST]
                                             [--datasets D ...] [--workers N]
                                             [--memory-limit GIGABYTES]
Writes RESULTS/diagnosis/MODEL.csv, with one row per task, and continues from the
tasks that are missing.
"""

import argparse
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import homogeneity_completeness_v_measure

from benchmark.evaluation import BATCH_SIZE, batches
from benchmark.models import MODELS, build
from benchmark.parallel import STALL_LIMIT, finished_tasks, run_tasks
from benchmark.preprocessing import segments
from benchmark.real_artifacts import DATASETS
from benchmark.runner import task_seeds

QUANTITIES = [
    "noise",
    "predicted_clusters",
    "classes",
    "homogeneity",
    "completeness",
    "microclusters",
    "mature",
    "active",
]
TASK_COLUMNS = ["dataset", "model", "run", "status", *QUANTITIES]


def state(model) -> dict:
    """Microclusters of a model of the MicroTEDAClus family; empty for the others."""
    core = getattr(model, "core", None)
    if core is None or not hasattr(core, "_count"):
        return {}
    n = core.microcluster_count()
    return {
        "microclusters": n,
        "mature": int((core._count[:n] >= 3).sum()),
        "active": int(core._active[:n].sum()),
    }


def replay(model, X, y) -> pd.DataFrame:
    rows = []
    for rows_of_batch in batches(len(X), BATCH_SIZE):
        Xb, yb = X[rows_of_batch], y[rows_of_batch]
        predicted = np.asarray(model.predict(Xb))
        clustered = predicted >= 0
        homogeneity, completeness, _ = (
            homogeneity_completeness_v_measure(yb[clustered], predicted[clustered])
            if clustered.sum() > 1
            else (np.nan, np.nan, np.nan)
        )
        rows.append(
            {
                "noise": 1.0 - clustered.mean(),
                "predicted_clusters": len(np.unique(predicted[clustered])),
                "classes": len(np.unique(yb)),
                "homogeneity": homogeneity,
                "completeness": completeness,
                **state(model),
            }
        )
        model.learn(Xb)
    return pd.DataFrame(rows)


def diagnose_task(dataset: str, model: str, run: int, hyperparameters: str) -> dict:
    """Replay one task; its row of the diagnosis."""
    warnings.simplefilter("ignore")
    row = {"dataset": dataset, "model": model, "run": run}
    try:
        _, (X, y) = segments(dataset, run)
        selected = json.loads(hyperparameters)
        built = build(MODELS[model], selected, task_seeds(dataset, model, run)[1])
        medians = replay(built, X, y).median(numeric_only=True)
        return row | {"status": "OK"} | medians.reindex(QUANTITIES).to_dict()
    except Exception as error:
        return row | {"status": f"ERROR: {error!r}"}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("results", type=Path)
    parser.add_argument("model", choices=list(MODELS))
    parser.add_argument("--runs", type=int, nargs=2, default=[0, 31])
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument(
        "--memory-limit",
        type=float,
        help="gigabytes of memory per worker; a task that needs more fails",
    )
    arguments = parser.parse_args()

    selected = pd.read_csv(arguments.results / "tasks.csv")
    selected = selected[
        (selected.status == "OK")
        & (selected.model == arguments.model)
        & selected.dataset.isin(arguments.datasets)
        & selected.run.between(*arguments.runs)
    ].sort_values(["run", "dataset"])
    out = arguments.results / "diagnosis"
    out.mkdir(exist_ok=True)
    tasks_file = out / f"{arguments.model}.csv"
    finished = finished_tasks(tasks_file)
    tasks = [
        (task.dataset, task.model, task.run, task.hyperparameters)
        for task in selected.itertuples()
        if (task.dataset, task.model, task.run) not in finished
    ]
    print(f"{len(tasks)} tasks to replay, {len(finished)} already finished", flush=True)
    run_tasks(
        diagnose_task,
        tasks,
        tasks_file,
        TASK_COLUMNS,
        arguments.workers,
        STALL_LIMIT,
        memory_limit=(
            None
            if arguments.memory_limit is None
            else int(arguments.memory_limit * 2**30)
        ),
    )


if __name__ == "__main__":
    main()
