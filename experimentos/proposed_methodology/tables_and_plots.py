#!/usr/bin/env python3
"""Tables and figures of the comparisons of the main experiment.

Each comparison sets a model in focus against each of the other models
(`editorial_config.COMPARISONS`); `statistic_analysis.py` computes the verdicts
that the tables mark.
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from editorial_config import (
    COMPARISONS,
    DATASET_ORDER,
    DATASET_SHORT_NAMES,
    effective_models,
    model_display_name,
)
from pfc_shared.figures import band
from pfc_shared.latex import marked, panel_table

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPOSITORY_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))

DIR_METRICS_NON_SCL = os.path.join(SCRIPT_DIR, "metrics", "non_scalability")
DIR_METRICS_SCL = os.path.join(SCRIPT_DIR, "metrics", "scalability")
DIR_STATS_BASE = os.path.join(SCRIPT_DIR, "statistical_analysis")

QUALITY_METRICS = ("ari_overall", "ari_steadystate", "ari_fifth", "ari_diff")
COST_METRICS = (
    "time_fit_cum_ms",
    "time_predict_cum_ms",
    "memory_max_bytes",
    "memory_steadystate_bytes",
)
SUMMARY_METRICS = ("mc_max", "mc_steadystate")
TITLES = {
    "ari_overall": "Overall ARI",
    "ari_steadystate": "Steady-State ARI",
    "ari_fifth": "ARI at the fifth batch",
    "ari_diff": "Maximum ARI drop",
    "time_fit_cum_ms": "Learning time [ms]",
    "time_predict_cum_ms": "Prediction time [ms]",
    "memory_max_bytes": "Maximum memory footprint [KiB]",
    "memory_steadystate_bytes": "Steady-state memory footprint [KiB]",
    "mc_max": "Maximum microcluster count",
    "mc_steadystate": "Steady-state microcluster count",
}


def shown(metric: str):
    """Text of a median: ARI with three decimals, time in ms, memory in KiB, counts whole."""
    if metric.startswith("ari"):
        return "{:.3f}".format
    if metric.startswith("memory"):
        return lambda value: f"{value / 1024:.0f}"
    return "{:.0f}".format


def output_dir(comparison) -> str:
    """Artifacts directory of the thesis chapter that presents a comparison."""
    return os.path.join(
        REPOSITORY_ROOT, "monografia", comparison.chapter, "artifacts", comparison.name
    )


def model_label(model: str) -> str:
    return f"\\modelname{{{model_display_name(model)}}}"


def statistics(comparison, metric: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(DIR_STATS_BASE, comparison.name, f"{metric}.csv"))


def run_text(stats: pd.DataFrame) -> str:
    """The runs over which the medians of a table are taken: those that every model finished."""
    runs = stats.groupby("dataset")["runs"].first()
    most = runs.max()
    exceptions = []
    for count in sorted(set(runs[runs < most]), reverse=True):
        names = [
            DATASET_SHORT_NAMES[dataset]
            for dataset in DATASET_ORDER
            if runs.get(dataset) == count
        ]
        listed = (
            names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
        )
        exceptions.append(f"{count} in {listed}")
    if not exceptions:
        return f"the {most} runs"
    return (
        f"the {most} runs, "
        + "; ".join(exceptions)
        + ", where a model has no result for a run"
    )


def panels(comparison, metrics) -> list:
    """A panel per metric: the model in focus, its count of verdicts, then the other models."""
    result = []
    for metric in metrics:
        stats = statistics(comparison, metric)
        focus, *others = effective_models(comparison, metric)
        text = shown(metric)
        of_focus = stats[stats.model == focus].set_index("dataset")
        counts = {
            dataset: "/".join(
                str((of_dataset.verdict == verdict).sum())
                for verdict in ("better", "same", "worse")
            )
            for dataset, of_dataset in stats[stats.model != focus].groupby("dataset")
        }
        rows = [
            (model_label(focus), of_focus["median"].to_dict(), text),
            ("Better/same/worse", counts, str),
        ]
        for model in others:
            of_model = stats[stats.model == model].set_index("dataset")
            rows.append(
                (
                    model_label(model),
                    {
                        dataset: marked(text(row["median"]), row["verdict"])
                        for dataset, row in of_model.iterrows()
                    },
                    str,
                )
            )
        result.append((TITLES[metric], rows))
    return result


def generate_tables(comparison, out_dir: str):
    columns = {dataset: DATASET_SHORT_NAMES[dataset] for dataset in DATASET_ORDER}
    subject = comparison.subject
    specs = [
        (
            "quality",
            QUALITY_METRICS,
            f"Clustering quality of {subject}",
            "",
        ),
        (
            "cost",
            COST_METRICS,
            f"Execution time and memory footprint of {subject}",
            "",
        ),
        (
            "microclusters",
            SUMMARY_METRICS,
            f"Microcluster count of {subject}",
            " \\modelname{StreamKM} keeps no microclusters.",
        ),
    ]
    for name, metrics, title, note in specs:
        runs = run_text(statistics(comparison, metrics[0]))
        panel_table(
            Path(out_dir) / f"{name}.tex",
            f"{title}: medians over {runs}.{note}",
            f"tab:{comparison.label}_{name}",
            columns,
            panels(comparison, metrics),
            short=title,
        )


def draw(ax, frame: pd.DataFrame, metric: str, x: str, comparison, scale=1.0):
    """Median and interquartile band of every model of a comparison along `x`."""
    for model in effective_models(comparison, metric):
        of_model = frame[frame["model"] == model].sort_values(x)
        if of_model.empty:
            continue
        band(
            ax,
            of_model[x].to_numpy(),
            of_model[f"{metric}_median"].to_numpy() / scale,
            of_model[f"{metric}_q1"].to_numpy() / scale,
            of_model[f"{metric}_q3"].to_numpy() / scale,
            model,
            model_display_name(model),
            model == comparison.focus,
        )


def dimension_axis(ax, dimensions):
    ax.set_xscale("log", base=2)
    ax.set_xticks(dimensions)
    ax.set_xticklabels([str(d) for d in dimensions])
    ax.set_xlabel("Dimension")


def save(fig, axes, path: str):
    """One legend above the panels, with the models of the first panel."""
    handles, labels = np.ravel(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def scalability_figures(comparison, out_dir: str):
    """ARI, then cost, of the models of a comparison in the scalability streams."""
    by_batch = pd.read_parquet(
        os.path.join(DIR_METRICS_SCL, "aggregated_by_batch.parquet")
    )
    by_dimension = pd.read_parquet(
        os.path.join(DIR_METRICS_SCL, "aggregated_by_dim.parquet")
    )
    by_batch = by_batch.assign(samples=by_batch["batch"] * 100)
    dimensions = sorted(by_dimension["dim"].unique())
    penultimate = by_batch["batch"].max() - 1
    os.makedirs(os.path.join(out_dir, "plots"), exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, dimension in zip(axes[0], (2, 64)):
        draw(ax, by_batch[by_batch["dim"] == dimension], "ari", "samples", comparison)
        ax.set_title(f"Along the stream, {dimension} dimensions")
        ax.set_xlabel("Samples")
        ax.set_ylabel("ARI")
    for ax, (batch, title) in zip(
        axes[1], ((4, "Fifth batch"), (penultimate, "Penultimate batch"))
    ):
        draw(ax, by_batch[by_batch["batch"] == batch], "ari", "dim", comparison)
        ax.set_title(title)
        ax.set_ylabel("ARI")
        dimension_axis(ax, dimensions)
    save(fig, axes, os.path.join(out_dir, "plots", "scalability_quality.png"))

    at_penultimate = by_batch[by_batch["batch"] == penultimate]
    specs = [
        (by_dimension, "time_fit_cum_ms", 1.0, "Learning time [ms]"),
        (by_dimension, "time_predict_cum_ms", 1.0, "Prediction time [ms]"),
        (by_dimension, "memory_max_bytes", 1024.0, "Maximum memory footprint [KiB]"),
        (
            at_penultimate,
            "memory_bytes",
            1024.0,
            "Steady-state memory footprint [KiB]",
        ),
        (by_dimension, "mc_max", 1.0, "Maximum microcluster count"),
        (at_penultimate, "microcluster_count", 1.0, "Steady-state microcluster count"),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(10, 10))
    for ax, (frame, metric, scale, ylabel) in zip(np.ravel(axes), specs):
        draw(ax, frame, metric, "dim", comparison, scale)
        ax.set_yscale("log")
        ax.set_ylabel(ylabel)
        dimension_axis(ax, dimensions)
    save(fig, axes, os.path.join(out_dir, "plots", "scalability_cost.png"))


def plot_internal_inspection(comparison, out_dir: str):
    df = pd.read_parquet(
        os.path.join(DIR_METRICS_NON_SCL, "aggregated_by_batch.parquet")
    )
    metrics = [
        "ari",
        "time_predict_ms",
        "time_fit_ms",
        "memory_bytes",
        "microcluster_count",
    ]
    os.makedirs(out_dir, exist_ok=True)

    for dataset in df["dataset"].unique():
        df_ds = df[df["dataset"] == dataset]
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.flatten()

        for i, metric in enumerate(metrics):
            ax = axes[i]
            draw(ax, df_ds, metric, "batch", comparison)
            ax.set_title(metric.replace("_", " ").title())
            ax.set_xlabel("Batch Index")
            if metric != "ari":
                ax.set_yscale("log")
            if i == 0:
                ax.legend(loc="best", fontsize=8)

        axes[-1].axis("off")
        plt.suptitle(f"Internal Inspection: {dataset}", y=1.02)
        plt.tight_layout()
        fig.savefig(
            os.path.join(out_dir, f"{dataset}_grid.png"), dpi=150, bbox_inches="tight"
        )
        plt.close(fig)


def main():
    for comparison in COMPARISONS:
        out_dir = output_dir(comparison)
        os.makedirs(out_dir, exist_ok=True)
        generate_tables(comparison, out_dir)
        scalability_figures(comparison, out_dir)
        plot_internal_inspection(comparison, os.path.join(out_dir, "plots", "internal"))
        print(f"Generated tables and plots for: {comparison.name}")


if __name__ == "__main__":
    main()
