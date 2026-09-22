#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from editorial_config import (
    DATASET_ORDER,
    MODEL_COLORS,
    SUBSETS,
    dataset_display_name,
    effective_models,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPOSITORY_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))

DIR_METRICS_NON_SCL = os.path.join(SCRIPT_DIR, "metrics", "non_scalability")
DIR_METRICS_SCL = os.path.join(SCRIPT_DIR, "metrics", "scalability")
DIR_STATS_BASE = os.path.join(SCRIPT_DIR, "statistical_analysis")
DIR_OUT_BASE = os.path.join(
    REPOSITORY_ROOT, "monografia", "MethodologyReassessment", "artifacts"
)

COLORS = MODEL_COLORS

READABLE_METRICS = {
    "ari_overall": "Overall Adjusted Rand Index",
    "ari_steadystate": "Adjusted Rand Index at the Penultimate Batch",
    "ari_fifth": "Adjusted Rand Index at the Fifth Batch",
    "ari_diff": "Maximum ARI Drop",
    "time_predict_cum_ms": "Cumulative Prediction Time [ms]",
    "time_fit_cum_ms": "Cumulative Fitting Time [ms]",
    "memory_max_bytes": "Maximum Memory Footprint [B]",
    "memory_steadystate_bytes": "Steady-State Memory Footprint at the Penultimate Batch [B]",
    "mc_max": "Maximum Microcluster Count",
    "mc_steadystate": "Steady-State Microcluster Count at the Penultimate Batch",
}


def format_latex_cell(stat_cell: str) -> str:
    stat_cell_str = str(stat_cell)
    if stat_cell_str == "nan" or stat_cell_str == "-":
        return "-"

    parts = stat_cell_str.split(" [")
    base = parts[0]
    tags = parts[1].replace("]", "") if len(parts) > 1 else ""

    if "B" in tags and "W" in tags:
        return base
    elif "B" in tags:
        return f"\\textbf{{{base}}}"
    elif "W" in tags:
        return f"\\textcolor{{red}}{{{base}}}"
    return base


def format_model_name(model: str) -> str:
    return f"\\modelname{{{model}}}"


def format_dataset_name(dataset: str) -> str:
    escaped_dataset = dataset_display_name(dataset).replace("_", "\\_")
    return f"\\datasetname{{{escaped_dataset}}}"


def generate_latex_tables(subset_name: str, subset_models: list, out_dir: str):
    dir_stats = os.path.join(DIR_STATS_BASE, subset_name)

    if not os.path.exists(dir_stats):
        return

    stat_files = [f for f in os.listdir(dir_stats) if f.endswith(".csv")]

    for stat_file in stat_files:
        metric = stat_file.replace("_statistical_analysis.csv", "")
        table_models = effective_models(subset_name, metric)
        df_stat = pd.read_csv(os.path.join(dir_stats, stat_file), index_col="dataset")
        dataset_order = {dataset: index for index, dataset in enumerate(DATASET_ORDER)}
        df_stat = df_stat.sort_index(
            key=lambda index: index.map(
                lambda dataset: dataset_order.get(dataset, len(DATASET_ORDER))
            )
        )
        readable_name = READABLE_METRICS.get(metric, metric.replace("_", " ").title())

        latex_lines = [
            "\\begin{table}[htbp]",
            "\\centering",
            f"\\caption{{Statistical analysis and (Q1, Q3) for the {readable_name}.}}",
            "\\resizebox{\\textwidth}{!}{%",
            "\\begin{tabular}{l " + "c " * len(table_models) + "}",
            "\\toprule",
            "Dataset & "
            + " & ".join(format_model_name(model) for model in table_models)
            + " \\\\",
            "\\midrule",
        ]

        for dataset in df_stat.index:
            tex_dataset = format_dataset_name(dataset)
            row_str = f"{tex_dataset}"
            for model in table_models:
                if model not in df_stat.columns:
                    row_str += " & -"
                    continue

                stat_cell = df_stat.loc[dataset, model]
                if np.asarray(pd.isna(stat_cell)).all():
                    row_str += " & -"
                    continue

                row_str += f" & {format_latex_cell(str(stat_cell))}"
            latex_lines.append(row_str + " \\\\")

        latex_lines.extend(
            [
                "\\bottomrule",
                "\\end{tabular}",
                "}",
                f"\\label{{tab:{subset_name}_{metric}}}",
                "\\end{table}",
            ]
        )

        with open(os.path.join(out_dir, f"{metric}.tex"), "w") as f:
            f.write("\n".join(latex_lines))
            f.write("\n")


def plot_scalability(subset_name: str, subset_models: list, out_dir: str):
    df = pd.read_parquet(os.path.join(DIR_METRICS_SCL, "aggregated_by_batch.parquet"))
    metrics = [
        "ari",
        "time_predict_ms",
        "time_fit_ms",
        "memory_bytes",
        "microcluster_count",
    ]

    for dim in [2, 64]:
        df_dim = df[df["dim"] == dim].copy()
        df_dim["samples"] = df_dim["batch"] * 100

        for metric in metrics:
            plot_models = effective_models(subset_name, metric)

            fig, ax = plt.subplots(figsize=(6, 4))
            is_log = metric != "ari"

            for model in plot_models:
                df_m = df_dim[df_dim["model"] == model].sort_values("samples")
                if df_m.empty:
                    continue

                med = df_m[f"{metric}_median"].to_numpy()
                q1 = df_m[f"{metric}_q1"].to_numpy()
                q3 = df_m[f"{metric}_q3"].to_numpy()

                ax.plot(
                    df_m["samples"],
                    med,
                    label=model,
                    color=COLORS[model],
                    linewidth=1.5,
                )
                ax.fill_between(
                    df_m["samples"], q1, q3, color=COLORS[model], alpha=0.15
                )

            ax.set_xlabel("Number of Samples")
            ax.set_ylabel(metric.replace("_", " ").title())
            if is_log:
                ax.set_yscale("log")
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.25), ncol=4)
            plt.tight_layout()
            fig.savefig(
                os.path.join(out_dir, f"sample_scalability_{metric}_dim{dim}.png"),
                dpi=300,
            )
            plt.close(fig)

    # ---- Feature-scalability ----
    df_dim_agg = pd.read_parquet(
        os.path.join(DIR_METRICS_SCL, "aggregated_by_dim.parquet")
    )
    max_batch = df["batch"].max()
    penultimate = max_batch - 1

    feature_scalability_specs = [
        # (filename, metric_col, source, batch_idx, is_log, ylabel)
        (f"feature_scalability_ari_batch4.png", "ari", "batch", 4, False, "ARI"),
        (
            f"feature_scalability_ari_steadystate.png",
            "ari",
            "batch",
            penultimate,
            False,
            "ARI",
        ),
        (
            "feature_scalability_time_predict_ms_cumulative.png",
            "time_predict_cum_ms",
            "dim",
            None,
            True,
            "Cumulative Prediction Time [ms]",
        ),
        (
            "feature_scalability_time_fit_ms_cumulative.png",
            "time_fit_cum_ms",
            "dim",
            None,
            True,
            "Cumulative Fitting Time [ms]",
        ),
        (
            "feature_scalability_memory_bytes_max.png",
            "memory_max_bytes",
            "dim",
            None,
            True,
            "Maximum Memory Footprint [B]",
        ),
        (
            f"feature_scalability_memory_bytes_steadystate.png",
            "memory_bytes",
            "batch",
            penultimate,
            True,
            "Memory Footprint [B]",
        ),
        (
            "feature_scalability_microcluster_count_max.png",
            "mc_max",
            "dim",
            None,
            True,
            "Maximum Microcluster Count",
        ),
        (
            f"feature_scalability_microcluster_count_steadystate.png",
            "microcluster_count",
            "batch",
            penultimate,
            True,
            "Steady State Microcluster Count",
        ),
    ]

    for (
        fname,
        metric_col,
        src_type,
        batch_idx,
        is_log,
        ylabel,
    ) in feature_scalability_specs:
        if src_type == "dim":
            df_src = df_dim_agg
        else:
            df_src = df[df["batch"] == batch_idx]

        models = effective_models(subset_name, metric_col)

        fig, ax = plt.subplots(figsize=(6, 4))
        for model in models:
            df_m = df_src[df_src["model"] == model].sort_values("dim")
            if df_m.empty:
                continue

            med = df_m[f"{metric_col}_median"].to_numpy()
            q1 = df_m[f"{metric_col}_q1"].to_numpy()
            q3 = df_m[f"{metric_col}_q3"].to_numpy()

            ax.plot(
                df_m["dim"],
                med,
                label=model,
                color=COLORS[model],
                linewidth=1.5,
            )
            ax.fill_between(
                df_m["dim"],
                q1,
                q3,
                color=COLORS[model],
                alpha=0.15,
            )

        ax.set_xlabel("Feature Dimension")
        ax.set_ylabel(ylabel)
        if is_log:
            ax.set_yscale("log")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.25), ncol=4)
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, fname), dpi=300)
        plt.close(fig)


def plot_internal_inspection(subset_name: str, subset_models: list, out_dir: str):
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

    for dataset in df["dataset"].unique():
        df_ds = df[df["dataset"] == dataset]
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.flatten()

        for i, metric in enumerate(metrics):
            ax = axes[i]
            is_log = metric != "ari"

            plot_models = effective_models(subset_name, metric)

            for model in plot_models:
                df_m = df_ds[df_ds["model"] == model].sort_values("batch")
                if df_m.empty:
                    continue

                med = df_m[f"{metric}_median"].values
                q1 = df_m[f"{metric}_q1"].values
                q3 = df_m[f"{metric}_q3"].values

                ax.plot(
                    df_m["batch"], med, label=model, color=COLORS[model], linewidth=1.5
                )
                ax.fill_between(df_m["batch"], q1, q3, color=COLORS[model], alpha=0.15)

            ax.set_title(metric.replace("_", " ").title())
            ax.set_xlabel("Batch Index")
            if is_log:
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
    for subset_name, subset_models in SUBSETS.items():
        dir_tables = os.path.join(DIR_OUT_BASE, subset_name, "tables")
        dir_plots_scl = os.path.join(DIR_OUT_BASE, subset_name, "plots", "scalability")
        dir_plots_internal = os.path.join(
            DIR_OUT_BASE, subset_name, "plots", "internal"
        )

        for d in [dir_tables, dir_plots_scl, dir_plots_internal]:
            os.makedirs(d, exist_ok=True)

        generate_latex_tables(subset_name, subset_models, dir_tables)
        plot_scalability(subset_name, subset_models, dir_plots_scl)
        plot_internal_inspection(subset_name, subset_models, dir_plots_internal)
        print(f"Generated tables and plots for subset: {subset_name}")


if __name__ == "__main__":
    main()
