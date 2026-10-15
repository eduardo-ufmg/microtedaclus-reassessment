"""Tables and figures of the experiment on the real datasets.

Written under monografia/RealData/artifacts/ from the results of
`run_real_experiment.sh`. A dataset is analysed over the runs that every model
finished on it, so a task that fails removes its run from its dataset only.

Usage (from experimentos/modern_methodology):
    python -m benchmark.real_artifacts [--results DIR] [--out DIR]
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from benchmark import analysis
from benchmark.artifacts import (
    GROUPS,
    TITLES,
    _bands,
    _save,
    at_the_ends,
    tier_panels,
)
from benchmark.datasets import REAL_DATASETS
from benchmark.display import columns, model_label
from benchmark.evaluation import BATCH_SIZE
from benchmark.models import MODELS
from benchmark.runner import batches_file
from benchmark.tuning import TUNING_LIMIT
from pfc_shared.latex import panel_table

DATASETS = list(REAL_DATASETS)
OUT = analysis.ROOT.parent.parent / "monografia" / "RealData" / "artifacts"
# Change points of the abrupt INSECTS stream, in samples from its start (Souza et al., 2020)
INSECTS_ABRUPT_CHANGES = (14352, 19500, 33240, 38682, 39510)


def complete_by_dataset(results, models):
    """The results of the runs that every model finished on each dataset, and their number."""
    kept, runs = [], {}
    for dataset in DATASETS:
        of_dataset = analysis.on_complete_runs(results, models, [dataset])
        kept.append(of_dataset)
        runs[dataset] = of_dataset.run.nunique()
    return pd.concat(kept), runs


def run_text(runs: dict) -> str:
    low, high = min(runs.values()), max(runs.values())
    return (
        f"{low} runs"
        if low == high
        else f"{low} to {high} runs, depending on the dataset"
    )


def quality_table(results, models, runs, out):
    panel_table(
        out / "quality.tex",
        f"Quality of the models on the real datasets: medians over {run_text(runs)}. {GROUPS}",
        "tab:real_quality",
        columns(DATASETS),
        tier_panels(results, analysis.QUALITY_METRICS, models, DATASETS),
    )


def summary_table(results, models, runs, out):
    panel_table(
        out / "summary_size.tex",
        "Number of structures in the summary that each model keeps of the stream, on the real "
        f"datasets: medians over {run_text(runs)}. {GROUPS}",
        "tab:real_summary_size",
        columns(DATASETS),
        tier_panels(results, analysis.SUMMARY_METRICS, models, DATASETS),
    )


def ends_table(results_dir, results, models, out):
    """Where the selected values lie in the widened ranges."""
    every_run = tuple(sorted(results.run.unique()))
    ranges = analysis.search_ranges(results_dir, every_run, tuple(DATASETS))
    ends = at_the_ends(results, ranges)
    percent = "{:.0f}".format
    panels = []
    for end, title in (("lower", "At the lower end"), ("upper", "At the upper end")):
        rows = []
        for model in models:
            for name in MODELS[model].HYPERPARAMETERS:
                of_name = ends[(ends.model == model) & (ends.hyperparameter == name)]
                values = (100 * of_name.set_index("dataset")[end]).to_dict()
                escaped = name.replace("_", r"\_")
                rows.append(
                    (rf"{model_label(model)}, \texttt{{{escaped}}}", values, percent)
                )
        panels.append((title, rows))
    panel_table(
        out / "ends.tex",
        "Percentage of the runs in which the selected value of a hyperparameter lies within a "
        "twentieth of the width of its widened search range from one of its ends, in the scale "
        "of the search.",
        "tab:real_ends",
        columns(DATASETS),
        panels,
        first_column="Model and hyperparameter",
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
        _bands(ax, (np.arange(values.shape[1]) + 1) * BATCH_SIZE, values, model)
    for change in INSECTS_ABRUPT_CHANGES:
        ax.axvline(change - TUNING_LIMIT, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Samples of the evaluation segment")
    ax.set_ylabel("ARI")
    _save(fig, [ax], "insects_abrupt.png", columns_of_legend=6, directory=out)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results", type=Path, default=analysis.REAL)
    parser.add_argument("--out", type=Path, default=OUT)
    arguments = parser.parse_args()

    results = analysis.run_level(arguments.results)
    models = [m for m in MODELS if m in set(results.model)]
    complete, runs = complete_by_dataset(results, models)
    print(f"{len(models)} models; complete runs by dataset: {runs}")

    arguments.out.mkdir(parents=True, exist_ok=True)
    quality_table(complete, models, runs, arguments.out)
    summary_table(complete, models, runs, arguments.out)
    ends_table(arguments.results, complete, models, arguments.out)
    abrupt_runs = sorted(complete[complete.dataset == "insects_abrupt"].run.unique())
    insects_abrupt_figure(arguments.results, models, abrupt_runs, arguments.out)


if __name__ == "__main__":
    main()
