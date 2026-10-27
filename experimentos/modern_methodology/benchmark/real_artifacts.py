"""Tables and figures of the experiment on the real datasets.

Written under monografia/RealData/artifacts/ from the results of `run_real_main.sh`.
A task fails when its model exceeds the limit of its summary or of its memory. A
model that failed in more than half of the runs of a dataset is left out of that
dataset, and the other models are compared over the runs that all of them finished
on it, so a failed task removes its run from its dataset only.

Usage (from experimentos/modern_methodology):
    python -m benchmark.real_artifacts [--results DIR] [--timing DIR] [--memory DIR]
                                       [--out DIR]
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from benchmark import analysis, real_data
from benchmark.artifacts import (
    SUBJECT,
    _bands,
    _in_kib,
    _in_ms,
    _save,
    complete_by_dataset,
    cost_by_dataset,
    failure_table,
    focus_panels,
    run_text,
)
from benchmark.datasets import REAL_DATASETS, load_stream
from benchmark.display import REAL_SHORT_NAMES, columns
from benchmark.evaluation import BATCH_SIZE
from benchmark.models import MODELS
from benchmark.runner import RUNS, batches_file
from benchmark.tuning import TUNING_LIMIT, split
from pfc_shared.latex import panel_table, write_table

DATASETS = list(REAL_DATASETS)
# Name macro of the thesis, reference and order of every dataset
DESCRIPTION = {
    "insects_abrupt": (r"\InsectsAbrupt", "souza2020challenges", "as recorded"),
    "insects_incremental": (
        r"\InsectsIncremental",
        "souza2020challenges",
        "as recorded",
    ),
    "insects_incremental_reoccurring": (
        r"\InsectsReoccurring",
        "souza2020challenges",
        "as recorded",
    ),
    "gas_drift": (
        r"\GasDrift",
        "vergara2012chemical",
        "periods in order, each shuffled",
    ),
    "kddcup99": (r"\KDDCup", "kddcup1999", "shuffled"),
    "covertype": (r"\Covertype", "blackard1999comparative", "as recorded"),
    "pendigits": (r"\Pendigits", "alpaydin1996pen", "shuffled"),
    "shuttle": (r"\Shuttle", "statlog_shuttle", "shuffled"),
}
SINGLE_BATCH_METRICS = ("ari_steadystate",)  # it reads the penultimate batch
OUT = analysis.ROOT.parent.parent / "monografia" / "RealData" / "artifacts"
# Change points of the abrupt INSECTS stream, in samples from its start (Souza et al., 2020)
INSECTS_ABRUPT_CHANGES = (14352, 19500, 33240, 38682, 39510)


def recorded_labels(dataset):
    """Labels of the stream in the order of its files, before any shuffle."""
    if dataset == "gas_drift":
        return real_data.gas_drift()[1]
    return REAL_DATASETS[dataset].generate(None, None)[1]


def single_class(labels) -> np.ndarray:
    """Whether each batch of the evaluation segment holds a single class."""
    (_, _), (_, y) = split(labels[:, None], labels)
    n = len(y) // BATCH_SIZE
    batches = y[: n * BATCH_SIZE].reshape(n, BATCH_SIZE)
    return np.array([len(np.unique(batch)) == 1 for batch in batches])


def single_class_shares(
    datasets=DATASETS, runs=range(RUNS[0], RUNS[1] + 1)
) -> pd.DataFrame:
    """Per dataset: the share of single-class batches in the recorded order and in the
    evaluated streams, and the share of runs whose penultimate batch holds one class."""
    rows = {}
    for dataset in datasets:
        evaluated = [single_class(load_stream(dataset, run)[1]) for run in runs]
        rows[dataset] = {
            "recorded": single_class(recorded_labels(dataset)).mean(),
            "evaluated": np.mean([batches.mean() for batches in evaluated]),
            "penultimate": np.mean([batches[-2] for batches in evaluated]),
        }
    return pd.DataFrame(rows).T


def single_class_table(shares, out):
    percent = "{:.1f}".format
    rows = [
        ("Recorded order", (100 * shares["recorded"]).to_dict(), percent),
        ("Streams evaluated", (100 * shares["evaluated"]).to_dict(), percent),
    ]
    penultimate = [
        (
            "Penultimate batch of a single class",
            (100 * shares["penultimate"]).to_dict(),
            "{:.0f}".format,
        )
    ]
    panel_table(
        out / "single_class.tex",
        "Batches of a single class in the evaluation segment of the real datasets, in the order "
        "of the files and in the streams evaluated, and runs whose penultimate batch holds a "
        "single class: percentages.",
        "tab:real_single_class",
        columns(DATASETS),
        [("Batches of a single class", rows), ("Runs", penultimate)],
        first_column="",
        short="Batches of a single class in the real datasets",
    )


def dataset_table(out):
    lines = [
        r"Dataset & Abbreviation & Samples & Features & Classes & Order \\",
        r"\midrule",
    ]
    for dataset in DATASETS:
        X, y = load_stream(dataset, 0)
        macro, reference, order = DESCRIPTION[dataset]
        lines.append(
            rf"{macro} \cite{{{reference}}} & {REAL_SHORT_NAMES[dataset]} & {len(X):,} & "
            rf"{X.shape[1]} & {len(np.unique(y))} & {order} \\"
        )
    write_table(
        out / "datasets.tex",
        "The real datasets: their number of samples, the features that the experiment uses, "
        "the number of classes, and the order of their streams.",
        "tab:real_datasets",
        "l l r r r l",
        lines,
        "The real datasets",
    )


def quality_table(results, models, runs, excluded, path, label, subject=SUBJECT):
    """The first of `models` in focus; the single-batch metrics are left out of the datasets
    in `excluded`."""
    panels = []
    for metric in analysis.QUALITY_METRICS:
        datasets = [
            d
            for d in DATASETS
            if not (metric in SINGLE_BATCH_METRICS and d in excluded)
        ]
        panels += focus_panels(results, (metric,), models, datasets)
    title = f"Clustering quality of {subject} on the real datasets"
    panel_table(
        path,
        f"{title}: medians over {run_text(runs)}. A dash marks a value left out: the "
        "Steady-State ARI of a stream whose penultimate batch holds a single class, a model "
        "that failed in more than half of the runs of a dataset, and every model of a dataset "
        r"that leaves \MicroTEDAClusPlus{} out, where no comparison is made.",
        label,
        columns(DATASETS),
        panels,
        short=title,
    )


def cost_table(complete, models, without_focus, timing_dir, memory_dir, out):
    """Time and memory, per dataset over the runs that every model kept in it finished; no
    comparison in the datasets that leave the first of `models`, in focus, out."""
    timing, memory, runs = cost_by_dataset(
        complete, models, DATASETS, without_focus, timing_dir, memory_dir
    )
    title = f"Execution time and memory of {SUBJECT} on the real datasets"
    panel_table(
        out / "cost.tex",
        f"{title}: medians over {run_text(runs)}. A dash marks a model left out of a dataset, "
        r"and every model of a dataset that leaves \MicroTEDAClusPlus{} out.",
        "tab:real_cost",
        columns(DATASETS),
        focus_panels(timing, analysis.TIME_METRICS, models, DATASETS, _in_ms)
        + focus_panels(memory, analysis.MEMORY_METRICS, models, DATASETS, _in_kib),
        short=title,
    )


def insects_abrupt_figure(results_dir, models, runs, out):
    """Batch ARI along the abrupt INSECTS stream, with its change points."""
    fig, ax = plt.subplots(figsize=(12, 4))
    for model in models:
        values = np.vstack(
            [
                pd.read_csv(batches_file(results_dir, "insects_abrupt", model, run))[
                    "ari"
                ].to_numpy()
                for run in runs
            ]
        )
        _bands(
            ax, (np.arange(values.shape[1]) + 1) * BATCH_SIZE, values, model, models[0]
        )
    for change in INSECTS_ABRUPT_CHANGES:
        ax.axvline(change - TUNING_LIMIT, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Samples of the evaluation segment")
    ax.set_ylabel("ARI")
    _save(fig, [ax], "insects_abrupt.png", columns_of_legend=6, directory=out)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results", type=Path, default=analysis.REAL_MAIN)
    parser.add_argument("--timing", type=Path, default=analysis.TIMING_REAL)
    parser.add_argument("--memory", type=Path, default=analysis.MEMORY_REAL)
    parser.add_argument("--out", type=Path, default=OUT)
    arguments = parser.parse_args()

    results = analysis.run_level(arguments.results)
    results = results[results.run.between(*RUNS)]
    tasks = pd.read_csv(arguments.results / "tasks.csv")
    tasks = tasks[tasks.run.between(*RUNS)]
    models = [m for m in MODELS if m in set(tasks.model)]
    complete, runs, without_focus = complete_by_dataset(
        results, tasks, models, DATASETS
    )
    print(f"{len(models)} models; complete runs by dataset: {runs}")
    print("datasets that leave the model in focus out:", without_focus)

    arguments.out.mkdir(parents=True, exist_ok=True)
    dataset_table(arguments.out)
    shares = single_class_shares()
    single_class_table(shares, arguments.out)
    excluded = set(shares.index[shares["penultimate"] > 0])
    print("single-batch metrics left out of:", sorted(excluded))
    failure_table(
        tasks,
        models,
        DATASETS,
        arguments.out / "failures.tex",
        "tab:real_failures",
        "Number of runs in which a model failed on a real dataset. Every failure is a "
        "summary that exceeded a quarter of the tuning segment.",
        "Failed runs on the real datasets",
    )
    quality_table(
        complete,
        models,
        runs,
        excluded,
        arguments.out / "quality.tex",
        "tab:real_quality",
    )
    abrupt = complete[complete.dataset == "insects_abrupt"]
    insects_abrupt_figure(
        arguments.results,
        [m for m in models if m in set(abrupt.model)],
        sorted(abrupt.run.unique()),
        arguments.out,
    )
    cost_table(
        complete,
        models,
        without_focus,
        arguments.timing,
        arguments.memory,
        arguments.out,
    )


if __name__ == "__main__":
    main()
