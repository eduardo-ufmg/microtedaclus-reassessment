#!/usr/bin/env python3
"""Tables and figures of the comparisons of the main experiment.

Each comparison sets a model in focus against each of the other models
(`editorial_config.COMPARISONS`); `statistic_analysis.py` computes the verdicts
that the tables mark.
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from editorial_config import (
    COMPARISONS,
    DATASET_ORDER,
    DATASET_SHORT_NAMES,
    effective_models,
    model_display_name,
)
from pfc_shared.figures import band
from pfc_shared.latex import focus_rows, panel_table

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPOSITORY_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))

DIR_METRICS_NON_SCL = os.path.join(SCRIPT_DIR, "metrics", "non_scalability")
DIR_STATS_BASE = os.path.join(SCRIPT_DIR, "statistical_analysis")

QUALITY_METRICS = ("ari_overall", "ari_steadystate")
COST_METRICS = (
    "time_fit_cum_ms",
    "time_predict_cum_ms",
    "memory_max_bytes",
    "memory_steadystate_bytes",
)
TITLES = {
    "ari_overall": "Overall ARI",
    "ari_steadystate": "Steady-State ARI",
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


def statistics(comparison, metric: str, prefix: str = "") -> pd.DataFrame:
    return pd.read_csv(
        os.path.join(DIR_STATS_BASE, comparison.name, f"{prefix}{metric}.csv")
    )


def panels(comparison, metrics, prefix: str = "") -> list:
    """A panel per metric: the model in focus, its count of verdicts, then the other models."""
    result = []
    for metric in metrics:
        focus, *others = effective_models(comparison, metric)
        rows = focus_rows(
            statistics(comparison, metric, prefix),
            focus,
            others,
            model_label,
            shown(metric),
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
    ]
    for name, metrics, title, note in specs:
        panel_table(
            Path(out_dir) / f"{name}.tex",
            f"{title}: medians over the runs of the main experiment.{note}",
            f"tab:{comparison.label}_{name}",
            columns,
            panels(comparison, metrics),
            short=title,
        )
    dimensions = statistics(comparison, QUALITY_METRICS[0], "scalability_")["dataset"]
    title = f"Clustering quality of {subject} in the scalability streams"
    panel_table(
        Path(out_dir) / "scalability.tex",
        f"{title}: medians over the runs of the main experiment.",
        f"tab:{comparison.label}_scalability",
        {
            dataset: f"S{dataset.removeprefix('SCL-')}"
            for dataset in sorted(dimensions.unique(), key=lambda d: int(d[4:]))
        },
        panels(comparison, QUALITY_METRICS, "scalability_"),
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
        plot_internal_inspection(comparison, os.path.join(out_dir, "plots", "internal"))
        print(f"Generated tables and plots for: {comparison.name}")


if __name__ == "__main__":
    main()
