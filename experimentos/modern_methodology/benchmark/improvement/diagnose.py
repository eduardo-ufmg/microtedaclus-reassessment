"""What MicroTEDAClus+ does on the real datasets: a replay of its selected configurations.

Each task of the real-data experiment is replayed with the hyperparameters that
its search selected, on the same evaluation segment, and at every batch the
state and the prediction of the model are recorded: its microclusters, the
active ones, its macroclusters, the share of samples labelled as noise, the
share of the largest predicted cluster, and the homogeneity and completeness of
the prediction against the classes. Mini-batch k-means, replayed the same way,
is the reference.

Usage (from experimentos/modern_methodology):
    python -m benchmark.improvement.diagnose [--runs N] [--datasets D ...]
"""

import argparse
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, homogeneity_completeness_v_measure

from benchmark.analysis import REAL
from benchmark.evaluation import BATCH_SIZE, batches
from benchmark.improvement.candidates import model_class
from benchmark.models import MODELS, build
from benchmark.preprocessing import segments
from benchmark.runner import task_seeds

DATASETS = [
    "insects_abrupt",
    "insects_incremental",
    "insects_incremental_reoccurring",
    "gas_drift",
    "pendigits",
    "shuttle",
]


def replay(model, X, y, inspect=None):
    rows = []
    for rows_of_batch in batches(len(X), BATCH_SIZE):
        Xb, yb = X[rows_of_batch], y[rows_of_batch]
        predicted = np.asarray(model.predict(Xb))
        clustered = predicted >= 0
        sizes = np.bincount(predicted[clustered]) if clustered.any() else np.array([0])
        homogeneity, completeness, _ = (
            homogeneity_completeness_v_measure(yb[clustered], predicted[clustered])
            if clustered.sum() > 1
            else (np.nan, np.nan, np.nan)
        )
        row = {
            "ari": adjusted_rand_score(yb, predicted),
            "noise": 1.0 - clustered.mean(),
            "largest": sizes.max() / len(Xb),
            "predicted_clusters": len(np.unique(predicted[clustered])),
            "classes": len(np.unique(yb)),
            "homogeneity": homogeneity,
            "completeness": completeness,
        }
        if inspect:
            row |= inspect(model)
        rows.append(row)
        model.learn(Xb)
    return pd.DataFrame(rows)


def engine_state(model):
    core = model.core
    pred = core._pred
    return {
        "microclusters": core._n,
        "active": 0 if pred is None else len(pred["index"]),
        "macroclusters": core.n_macro_,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs", type=int, default=4)
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    arguments = parser.parse_args()
    warnings.simplefilter("ignore")

    tasks = pd.read_csv(REAL / "tasks.csv")
    tasks = tasks[tasks.status == "OK"]
    control = model_class("control")
    summary = []
    for dataset in arguments.datasets:
        for run in range(arguments.runs):
            _, (X, y) = segments(dataset, run)
            for name in ("MicroTEDAClusPlus", "MiniBatchKMeans"):
                task = tasks[
                    (tasks.dataset == dataset)
                    & (tasks.run == run)
                    & (tasks.model == name)
                ]
                if task.empty:
                    continue
                hp = json.loads(task.iloc[0].hyperparameters)
                if name == "MicroTEDAClusPlus":
                    model = control(
                        variance_limit=hp["variance_limit"],
                        decay=hp["decay"],
                        noise_m=hp["noise_distance"],
                    )
                    inspect = engine_state
                else:
                    model = build(MODELS[name], hp, task_seeds(dataset, name, run)[1])
                    inspect = None
                result = replay(model, X, y, inspect)
                summary.append(
                    {
                        "dataset": dataset,
                        "run": run,
                        "model": name,
                        **result.median(numeric_only=True).to_dict(),
                        "hyperparameters": hp,
                    }
                )
    table = pd.DataFrame(summary)
    pd.set_option("display.width", 250)
    columns = [
        "ari",
        "noise",
        "largest",
        "predicted_clusters",
        "classes",
        "homogeneity",
        "completeness",
        "microclusters",
        "active",
        "macroclusters",
    ]
    print(
        table.groupby(["dataset", "model"])[[c for c in columns if c in table]]
        .median()
        .round(3)
        .to_string()
    )
    return table


if __name__ == "__main__":
    main()
