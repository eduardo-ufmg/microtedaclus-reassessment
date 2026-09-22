#!/usr/bin/env python3
import glob
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon
from statsmodels.stats.multitest import multipletests

from editorial_config import DATASET_ORDER, MODEL_COLORS, dataset_display_name

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPOSITORY_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))

DIR_RESULTS = os.path.join(SCRIPT_DIR, "results_iterative")
DIR_PARAMS = os.path.join(SCRIPT_DIR, "results_params_iterative")

DIR_METRICS_NON_SCL = os.path.join(SCRIPT_DIR, "metrics_iterative", "non_scalability")

DIR_STATS = os.path.join(SCRIPT_DIR, "statistical_analysis_iterative")
DIR_OUT_BASE = os.path.join(
    REPOSITORY_ROOT, "monografia", "CorrectedModel", "artifacts"
)
DIR_TABLES = os.path.join(DIR_OUT_BASE, "tables")
DIR_PLOTS_INT = os.path.join(DIR_OUT_BASE, "plots", "internal")

for d in [
    DIR_METRICS_NON_SCL,
    DIR_STATS,
    DIR_TABLES,
    DIR_PLOTS_INT,
]:
    os.makedirs(d, exist_ok=True)

ITERATIVE_MODELS = ["MicroTEDAClus", "MicroTEDAClusCorrected"]
ALPHA = 0.05
CLIFFS_DELTA_THRESHOLD = 0.147

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

STATS_METRICS = {
    "ari_overall": {"dir": "max", "round": lambda x: np.round(x, 3), "fmt": "{:.3f}"},
    "ari_steadystate": {
        "dir": "max",
        "round": lambda x: np.round(x, 3),
        "fmt": "{:.3f}",
    },
    "ari_fifth": {"dir": "max", "round": lambda x: np.round(x, 3), "fmt": "{:.3f}"},
    "ari_diff": {"dir": "min", "round": lambda x: np.round(x, 3), "fmt": "{:.3f}"},
    "time_predict_cum_ms": {
        "dir": "min",
        "round": lambda x: np.round(x, -1),
        "fmt": "{:.1f}",
    },
    "time_fit_cum_ms": {
        "dir": "min",
        "round": lambda x: np.round(x, -1),
        "fmt": "{:.1f}",
    },
    "memory_max_bytes": {
        "dir": "min",
        "round": lambda x: np.round(x / 1024) * 1024,
        "fmt": "{:.0f}",
    },
    "memory_steadystate_bytes": {
        "dir": "min",
        "round": lambda x: np.round(x / 1024) * 1024,
        "fmt": "{:.0f}",
    },
    "mc_max": {"dir": "min", "round": lambda x: np.round(x), "fmt": "{:.0f}"},
    "mc_steadystate": {"dir": "min", "round": lambda x: np.round(x), "fmt": "{:.0f}"},
}


def aggregate_quantiles(
    df: pd.DataFrame, group_cols: list, val_cols: list
) -> pd.DataFrame:
    grouped = df.groupby(group_cols)[val_cols]
    return pd.concat(
        [
            grouped.median().rename(columns=lambda c: f"{c}_median"),
            grouped.quantile(0.25).rename(columns=lambda c: f"{c}_q1"),
            grouped.quantile(0.75).rename(columns=lambda c: f"{c}_q3"),
        ],
        axis=1,
    ).reset_index()


def compute_metrics():
    result_files = glob.glob(os.path.join(DIR_RESULTS, "*.csv"))
    non_scl_scalar, non_scl_batch = [], []

    for file in result_files:
        parts = os.path.basename(file).replace(".csv", "").split("_")
        run, model, dataset = int(parts[-1]), parts[-2], "_".join(parts[:-2])

        # Exclude scalability subsets other than SCL-32 and SCL-64
        if dataset.startswith("SCL-") and dataset not in ["SCL-32", "SCL-64"]:
            continue

        df = pd.read_csv(file)

        if not df[df["status"].str.startswith("ERROR", na=False)].empty or len(df) < 5:
            continue

        df["dataset"], df["model"], df["run"] = dataset, model, run
        df["time_predict_ms"] = df["time_predict_wall_ns"] / 1e6
        df["time_fit_ms"] = df["time_fit_wall_ns"] / 1e6

        non_scl_batch.append(
            df[
                [
                    "dataset",
                    "model",
                    "run",
                    "batch",
                    "ari",
                    "time_predict_ms",
                    "time_fit_ms",
                    "memory_bytes",
                    "microcluster_count",
                ]
            ]
        )
        ari, mem, mc = (
            df["ari"].to_numpy(),
            df["memory_bytes"].to_numpy(),
            df["microcluster_count"].to_numpy(),
        )
        non_scl_scalar.append(
            {
                "dataset": dataset,
                "model": model,
                "run": run,
                "ari_overall": np.median(ari),
                "ari_steadystate": ari[-2],
                "ari_fifth": ari[4],
                "ari_diff": np.max(ari) - ari[-2],
                "time_predict_cum_ms": df["time_predict_ms"].sum(),
                "time_fit_cum_ms": df["time_fit_ms"].sum(),
                "memory_max_bytes": np.max(mem),
                "memory_steadystate_bytes": mem[-2],
                "mc_max": np.max(mc),
                "mc_steadystate": mc[-2],
            }
        )

    val_cols = [
        "ari",
        "time_predict_ms",
        "time_fit_ms",
        "memory_bytes",
        "microcluster_count",
    ]

    df_non_scl = pd.DataFrame(non_scl_scalar)
    df_non_scl.to_parquet(
        os.path.join(DIR_METRICS_NON_SCL, "run_level.parquet"), index=False
    )
    aggregate_quantiles(
        df_non_scl, ["dataset", "model"], list(STATS_METRICS.keys())
    ).to_parquet(os.path.join(DIR_METRICS_NON_SCL, "aggregated.parquet"), index=False)
    aggregate_quantiles(
        pd.concat(non_scl_batch, ignore_index=True),
        ["dataset", "model", "batch"],
        val_cols,
    ).to_parquet(
        os.path.join(DIR_METRICS_NON_SCL, "aggregated_by_batch.parquet"),
        index=False,
    )


def run_statistical_analysis():
    path_in = os.path.join(DIR_METRICS_NON_SCL, "run_level.parquet")
    if not os.path.exists(path_in):
        return
    df_runs = pd.read_parquet(path_in)

    for metric, config in STATS_METRICS.items():
        if metric not in df_runs.columns:
            continue
        results = []
        for dataset in df_runs["dataset"].unique():
            df_ds = df_runs[df_runs["dataset"] == dataset].pivot(
                index="run", columns="model", values=metric
            )
            valid_models = [m for m in ITERATIVE_MODELS if m in df_ds.columns]

            if len(valid_models) < 2:
                continue
            df_ds = df_ds[valid_models].dropna()
            if df_ds.empty:
                continue

            df_disc = pd.DataFrame(
                config["round"](df_ds.values), columns=valid_models, index=df_ds.index
            )
            medians, q1s, q3s = (
                df_disc.median(),
                df_disc.quantile(0.25),
                df_disc.quantile(0.75),
            )

            best_model = (
                medians.idxmax() if config["dir"] == "max" else medians.idxmin()
            )
            worst_model = (
                medians.idxmin() if config["dir"] == "max" else medians.idxmax()
            )

            is_sig = False
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    if len(valid_models) == 2:
                        if (
                            wilcoxon(
                                df_disc[valid_models[0]], df_disc[valid_models[1]]
                            ).pvalue
                            < ALPHA
                        ):
                            is_sig = True
                    else:
                        if (
                            friedmanchisquare(
                                *[
                                    np.asarray(df_disc[m].to_numpy(), dtype=float)
                                    for m in valid_models
                                ]
                            ).pvalue
                            < ALPHA
                        ):
                            is_sig = True
                except ValueError:
                    pass

            best_tier, worst_tier = set([best_model]), set([worst_model])
            if not is_sig:
                best_tier, worst_tier = set(valid_models), set(valid_models)
            else:

                def process_tier(target_model, target_tier):
                    pvals, test_models = [], [
                        m for m in valid_models if m != target_model
                    ]
                    for m in test_models:
                        diff = df_disc[target_model].to_numpy() - df_disc[m].to_numpy()
                        pvals.append(
                            1.0
                            if np.all(diff == 0)
                            else wilcoxon(df_disc[target_model], df_disc[m]).pvalue
                        )
                    _, p_adj, _, _ = multipletests(pvals, alpha=ALPHA, method="holm")

                    for m, p in zip(test_models, p_adj):
                        delta = float(
                            np.mean(
                                np.sign(
                                    df_disc[target_model].values[:, None]
                                    - df_disc[m].to_numpy()[None, :]
                                )
                            )
                        )
                        if p >= ALPHA or abs(delta) < CLIFFS_DELTA_THRESHOLD:
                            target_tier.add(m)

                process_tier(best_model, best_tier)
                process_tier(worst_model, worst_tier)

            row_dict = {"dataset": dataset}
            for m in valid_models:
                tags = [
                    t
                    for t, cond in [("B", m in best_tier), ("W", m in worst_tier)]
                    if cond
                ]
                tag_str = f" [{','.join(tags)}]" if tags else ""
                row_dict[m] = (
                    f"{config['fmt'].format(medians[m])} ({config['fmt'].format(q1s[m])}, {config['fmt'].format(q3s[m])}){tag_str}"
                )

            results.append(row_dict)

        if results:
            pd.DataFrame(results).set_index("dataset").to_csv(
                os.path.join(DIR_STATS, f"{metric}_statistical_analysis.csv")
            )


def format_latex_cell(stat_cell: str) -> str:
    if str(stat_cell) in ("nan", "-"):
        return "-"
    parts = str(stat_cell).split(" [")
    base, tags = parts[0], parts[1].replace("]", "") if len(parts) > 1 else ""
    return (
        base
        if "B" in tags and "W" in tags
        else (
            f"\\textbf{{{base}}}"
            if "B" in tags
            else f"\\textcolor{{red}}{{{base}}}" if "W" in tags else base
        )
    )


def generate_artifacts():
    stat_files = [f for f in os.listdir(DIR_STATS) if f.endswith(".csv")]
    for stat_file in stat_files:
        metric = stat_file.replace("_statistical_analysis.csv", "")
        df_stat = pd.read_csv(os.path.join(DIR_STATS, stat_file), index_col="dataset")

        # Extend the imported DATASET_ORDER locally
        local_order = list(DATASET_ORDER) + ["SCL-32", "SCL-64"]
        ds_order = {d: i for i, d in enumerate(local_order)}

        df_stat = df_stat.sort_index(
            key=lambda idx: idx.map(lambda d: ds_order.get(d, len(DATASET_ORDER)))
        )

        latex_lines = [
            "\\begin{table}[htbp]",
            "\\centering",
            f"\\caption{{Statistical analysis and (Q1, Q3) for the {READABLE_METRICS.get(metric, metric.replace('_', ' ').title())}.}}",
            "\\resizebox{\\textwidth}{!}{%",
            "\\begin{tabular}{l " + "c " * len(ITERATIVE_MODELS) + "}",
            "\\toprule",
            "Dataset & "
            + " & ".join(f"\\modelname{{{m}}}" for m in ITERATIVE_MODELS)
            + " \\\\",
            "\\midrule",
        ]

        for dataset in df_stat.index:
            row_str = (
                f"\\datasetname{{{dataset_display_name(dataset).replace('_', '\\_')}}}"
            )
            for model in ITERATIVE_MODELS:
                cell = df_stat.loc[dataset, model] if model in df_stat.columns else "-"
                row_str += (
                    f" & {format_latex_cell(str(cell))}"
                    if not np.asarray(pd.isna(cell)).all()
                    else " & -"
                )
            latex_lines.append(row_str + " \\\\")

        latex_lines.extend(
            [
                "\\bottomrule",
                "\\end{tabular}",
                "}",
                f"\\label{{tab:iterative_{metric}}}",
                "\\end{table}",
            ]
        )
        with open(os.path.join(DIR_TABLES, f"{metric}.tex"), "w") as f:
            f.write("\n".join(latex_lines))

    metrics = [
        "ari",
        "time_predict_ms",
        "time_fit_ms",
        "memory_bytes",
        "microcluster_count",
    ]

    int_path = os.path.join(DIR_METRICS_NON_SCL, "aggregated_by_batch.parquet")
    if os.path.exists(int_path):
        df_int = pd.read_parquet(int_path)
        for dataset in df_int["dataset"].unique():
            df_ds = df_int[df_int["dataset"] == dataset]
            fig, axes = plt.subplots(2, 3, figsize=(15, 8))
            for i, metric in enumerate(metrics):
                ax = axes.flatten()[i]
                for model in ITERATIVE_MODELS:
                    df_m = df_ds[df_ds["model"] == model].sort_values("batch")
                    if not df_m.empty:
                        ax.plot(
                            df_m["batch"],
                            df_m[f"{metric}_median"],
                            label=model,
                            color=MODEL_COLORS[model],
                            linewidth=1.5,
                        )
                        ax.fill_between(
                            df_m["batch"],
                            df_m[f"{metric}_q1"],
                            df_m[f"{metric}_q3"],
                            color=MODEL_COLORS[model],
                            alpha=0.15,
                        )
                ax.set_title(metric.replace("_", " ").title())
                ax.set_xlabel("Batch Index")
                if metric != "ari":
                    ax.set_yscale("log")
                if i == 0:
                    ax.legend(loc="best", fontsize=8)
            axes.flatten()[-1].axis("off")
            plt.suptitle(f"Internal Inspection: {dataset}", y=1.02)
            plt.tight_layout()
            fig.savefig(
                os.path.join(DIR_PLOTS_INT, f"{dataset}_grid.png"),
                dpi=150,
                bbox_inches="tight",
            )
            plt.close(fig)


if __name__ == "__main__":
    compute_metrics()
    run_statistical_analysis()
    generate_artifacts()
