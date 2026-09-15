#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DIR_METRICS_NON_SCL = "./metrics/non_scalability"
DIR_METRICS_SCL = "./metrics/scalability"
DIR_STATS_BASE = "./statistical_analysis"
DIR_OUT_BASE = "./artifacts"

SUBSETS = {
    "all": [
        "CluStream",
        "DenStream",
        "MicroTEDAClus",
        "CEDAS",
        "TEDACloud",
        "AutoCloud",
        "StreamKM",
    ],
    "original-paper": ["MicroTEDAClus", "DenStream", "CluStream", "StreamKM"],
    "teda-micro-based": ["MicroTEDAClus", "AutoCloud", "CEDAS", "TEDACloud"],
}

COLORS = dict(
    zip(
        SUBSETS["all"],
        ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"],
    )
)

READABLE_METRICS = {
    "ari_overall": "Overall Adjusted Rand Index",
    "ari_penultimate": "Adjusted Rand Index at the Penultimate Batch",
    "ari_fifth": "Adjusted Rand Index at the Fifth Batch",
    "ari_diff": "Maximum ARI Drop",
    "time_predict_cum_ms": "Cumulative Prediction Time [ms]",
    "time_fit_cum_ms": "Cumulative Fitting Time [ms]",
    "memory_max_bytes": "Maximum Memory Footprint [B]",
    "memory_penultimate_bytes": "Steady-State Memory Footprint at the Penultimate Batch [B]",
    "mc_max": "Maximum Microcluster Count",
    "mc_penultimate": "Steady-State Microcluster Count at the Penultimate Batch",
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


def generate_latex_tables(subset_name: str, subset_models: list, out_dir: str):
    dir_stats = os.path.join(DIR_STATS_BASE, subset_name)

    if not os.path.exists(dir_stats):
        return

    stat_files = [f for f in os.listdir(dir_stats) if f.endswith(".csv")]

    for stat_file in stat_files:
        metric = stat_file.replace("_statistical_analysis.csv", "")
        df_stat = pd.read_csv(os.path.join(dir_stats, stat_file), index_col="dataset")
        readable_name = READABLE_METRICS.get(metric, metric.replace("_", " ").title())

        latex_lines = [
            "\\begin{table}[htbp]",
            "\\centering",
            "\\resizebox{\\textwidth}{!}{%",
            "\\begin{tabular}{l " + "c " * len(subset_models) + "}",
            "\\toprule",
            "Dataset & " + " & ".join(subset_models) + " \\\\",
            "\\midrule",
        ]

        for dataset in df_stat.index:
            tex_dataset = dataset.replace("_", "\\_")
            row_str = f"{tex_dataset}"
            for model in subset_models:
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
                f"\\caption{{Statistical analysis and (Q1, Q3) for the {readable_name}.}}",
                f"\\label{{tab:{subset_name}_{metric}}}",
                "\\end{table}",
            ]
        )

        with open(os.path.join(out_dir, f"{metric}.tex"), "w") as f:
            f.write("\n".join(latex_lines))


def plot_scalability(subset_models: list, out_dir: str):
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
            fig, ax = plt.subplots(figsize=(6, 4))
            is_log = metric != "ari"

            for model in subset_models:
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
            fig.savefig(os.path.join(out_dir, f"typeA_{metric}_dim{dim}.png"), dpi=300)
            plt.close(fig)

    max_batch = df["batch"].max()
    for b_idx in [10, max_batch - 1]:
        df_b = df[df["batch"] == b_idx]

        for metric in metrics:
            fig, ax = plt.subplots(figsize=(6, 4))
            is_log = metric != "ari"

            for model in subset_models:
                df_m = df_b[df_b["model"] == model].sort_values("dim")
                if df_m.empty:
                    continue

                med = df_m[f"{metric}_median"].values
                q1 = df_m[f"{metric}_q1"].values
                q3 = df_m[f"{metric}_q3"].values

                ax.plot(
                    df_m["dim"], med, label=model, color=COLORS[model], linewidth=1.5
                )
                ax.fill_between(df_m["dim"], q1, q3, color=COLORS[model], alpha=0.15)

            ax.set_xlabel("Feature Dimension")
            ax.set_ylabel(metric.replace("_", " ").title())
            if is_log:
                ax.set_yscale("log")
            ax.set_xscale("log", base=2)
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.25), ncol=4)
            plt.tight_layout()
            fig.savefig(
                os.path.join(out_dir, f"typeB_{metric}_batch{b_idx}.png"), dpi=300
            )
            plt.close(fig)


def plot_internal_inspection(subset_models: list, out_dir: str):
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

            for model in subset_models:
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
        plot_scalability(subset_models, dir_plots_scl)
        plot_internal_inspection(subset_models, dir_plots_internal)
        print(f"Generated tables and plots for subset: {subset_name}")


if __name__ == "__main__":
    main()
