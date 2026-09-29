#!/usr/bin/env python3
import argparse
import glob
import os
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon
from statsmodels.stats.multitest import multipletests

from editorial_config import ITERATIVE_DATASET_ORDER, dataset_display_name
from improved_config import (
    IMPROVEMENT_STAGES,
    IMPROVED_MODEL_COLORS,
    IMPROVED_MODEL_SHORT_HEADERS,
    STAGE_BY_ID,
    ImprovementStage,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPOSITORY_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))

DIR_RESULTS = os.path.join(SCRIPT_DIR, "results_improved")
DIR_PARAMS = os.path.join(SCRIPT_DIR, "results_params_improved")

DIR_METRICS = os.path.join(SCRIPT_DIR, "metrics_improved")
DIR_STATS = os.path.join(SCRIPT_DIR, "statistical_analysis_improved")
DIR_OUT_BASE = os.path.join(REPOSITORY_ROOT, "monografia", "ImprovedModel")

for d in [DIR_METRICS, DIR_STATS]:
    os.makedirs(d, exist_ok=True)

ALPHA = 0.05
CLIFFS_DELTA_THRESHOLD = 0.147

READABLE_METRICS = {
    "ari_overall": "Overall Adjusted Rand Index",
    "ari_steadystate": "Steady-State Adjusted Rand Index",
    "ari_fifth": "Adjusted Rand Index at the Fifth Batch",
    "ari_diff": "Maximum ARI Drop",
    "time_predict_cum_ms": "Cumulative Prediction Time [ms]",
    "time_fit_cum_ms": "Cumulative Fitting Time [ms]",
    "memory_max_bytes": "Maximum Memory Footprint [B]",
    "memory_steadystate_bytes": "Steady-State Memory Footprint [B]",
    "mc_max": "Maximum Microcluster Count",
    "mc_steadystate": "Steady-State Microcluster Count",
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


def parse_result_filename(filename: str):
    """Robustly parse dataset, model tag, and run index from result CSV filename."""
    basename = os.path.basename(filename)
    if not basename.endswith(".csv"):
        return None
    name = basename[:-4]
    sorted_datasets = sorted(ITERATIVE_DATASET_ORDER, key=len, reverse=True)
    for ds in sorted_datasets:
        prefix = ds + "_"
        if name.startswith(prefix):
            remainder = name[len(prefix) :]
            if "_" in remainder:
                model, run_str = remainder.rsplit("_", 1)
                try:
                    return ds, model, int(run_str)
                except ValueError:
                    pass
    return None


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
    records_scalar, records_batch = [], []

    for file in result_files:
        parsed = parse_result_filename(file)
        if parsed is None:
            continue
        dataset, model, run = parsed

        try:
            df = pd.read_csv(file)
        except Exception:
            continue

        if not df[df["status"].str.startswith("ERROR", na=False)].empty or len(df) < 5:
            continue

        df["dataset"], df["model"], df["run"] = dataset, model, run
        df["time_predict_ms"] = df["time_predict_wall_ns"] / 1e6
        df["time_fit_ms"] = df["time_fit_wall_ns"] / 1e6

        records_batch.append(
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
        records_scalar.append(
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

    df_scalar = pd.DataFrame(records_scalar)
    df_scalar.to_parquet(os.path.join(DIR_METRICS, "run_level.parquet"), index=False)
    aggregate_quantiles(
        df_scalar, ["dataset", "model"], list(STATS_METRICS.keys())
    ).to_parquet(os.path.join(DIR_METRICS, "aggregated.parquet"), index=False)
    aggregate_quantiles(
        pd.concat(records_batch, ignore_index=True),
        ["dataset", "model", "batch"],
        val_cols,
    ).to_parquet(
        os.path.join(DIR_METRICS, "aggregated_by_batch.parquet"),
        index=False,
    )


def run_statistical_analysis(stage: ImprovementStage):
    path_in = os.path.join(DIR_METRICS, "run_level.parquet")
    if not os.path.exists(path_in):
        return
    df_runs = pd.read_parquet(path_in)

    stage_stats_dir = os.path.join(DIR_STATS, stage.stage_id)
    os.makedirs(stage_stats_dir, exist_ok=True)

    for metric, config in STATS_METRICS.items():
        if metric not in df_runs.columns:
            continue
        results = []
        for dataset in stage.datasets:
            df_ds_all = df_runs[df_runs["dataset"] == dataset]
            if df_ds_all.empty:
                continue

            df_pivot = df_ds_all.pivot(index="run", columns="model", values=metric)
            valid_models = [
                m
                for m in stage.models
                if m in df_pivot.columns and not df_pivot[m].isna().all()
            ]

            row_dict = {"dataset": dataset}

            if len(valid_models) < 2:
                for m in stage.models:
                    if m in valid_models and not df_pivot[m].dropna().empty:
                        s = df_pivot[m].dropna()
                        med, q1, q3 = s.median(), s.quantile(0.25), s.quantile(0.75)
                        row_dict[m] = (
                            f"{config['fmt'].format(med)} ({config['fmt'].format(q1)}, {config['fmt'].format(q3)})"
                        )
                    else:
                        row_dict[m] = "-"
                results.append(row_dict)
                continue

            df_ds = df_pivot[valid_models].dropna()
            if df_ds.empty or len(df_ds) < 2:
                for m in stage.models:
                    row_dict[m] = "-"
                results.append(row_dict)
                continue

            df_disc = pd.DataFrame(
                config["round"](df_ds.values), columns=valid_models, index=df_ds.index
            )
            medians = df_disc.median()
            q1s = df_disc.quantile(0.25)
            q3s = df_disc.quantile(0.75)

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
                        diff = (
                            df_disc[valid_models[0]].to_numpy()
                            - df_disc[valid_models[1]].to_numpy()
                        )
                        if not np.all(diff == 0):
                            pval = wilcoxon(
                                df_disc[valid_models[0]].to_numpy(),
                                df_disc[valid_models[1]].to_numpy(),
                            ).pvalue
                            if pval < ALPHA:
                                is_sig = True
                    else:
                        pval = friedmanchisquare(
                            *[
                                np.asarray(df_disc[m].to_numpy(), dtype=float)
                                for m in valid_models
                            ]
                        ).pvalue
                        if pval < ALPHA:
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

            for m in stage.models:
                if m in valid_models:
                    tags = [
                        t
                        for t, cond in [("B", m in best_tier), ("W", m in worst_tier)]
                        if cond
                    ]
                    tag_str = f" [{','.join(tags)}]" if tags else ""
                    row_dict[m] = (
                        f"{config['fmt'].format(medians[m])} ({config['fmt'].format(q1s[m])}, {config['fmt'].format(q3s[m])}){tag_str}"
                    )
                else:
                    row_dict[m] = "-"

            results.append(row_dict)

        if results:
            df_out = pd.DataFrame(results).set_index("dataset")
            df_out.to_csv(
                os.path.join(stage_stats_dir, f"{metric}_statistical_analysis.csv")
            )


def format_latex_cell(cell_val: str) -> str:
    if not isinstance(cell_val, str) or "[" not in cell_val:
        return cell_val
    parts = cell_val.split(" [")
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


def generate_artifacts(stage: ImprovementStage):
    stage_dir = os.path.join(DIR_OUT_BASE, stage.artifact_dir)
    tables_dir = os.path.join(stage_dir, "tables")
    plots_dir = os.path.join(stage_dir, "plots", "internal")
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    stage_stats_dir = os.path.join(DIR_STATS, stage.stage_id)
    if not os.path.exists(stage_stats_dir):
        return

    stat_files = [f for f in os.listdir(stage_stats_dir) if f.endswith(".csv")]
    for stat_file in stat_files:
        metric = stat_file.replace("_statistical_analysis.csv", "")
        df_stat = pd.read_csv(
            os.path.join(stage_stats_dir, stat_file), index_col="dataset"
        )

        latex_lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            f"\\caption{{Statistical analysis for the {READABLE_METRICS.get(metric, metric.replace('_', ' ').title())}.}}",
            r"\resizebox{\textwidth}{!}{%",
            r"\begin{tabular}{l " + "c " * len(stage.models) + "}",
            r"\toprule",
            "Dataset & "
            + " & ".join(IMPROVED_MODEL_SHORT_HEADERS.get(m, m) for m in stage.models)
            + r" \\",
            r"\midrule",
        ]

        for dataset in stage.datasets:
            if dataset not in df_stat.index:
                continue
            row_str = (
                f"\\datasetname{{{dataset_display_name(dataset).replace('_', '\\_')}}}"
            )
            for model in stage.models:
                cell = df_stat.loc[dataset, model] if model in df_stat.columns else "-"
                row_str += (
                    f" & {format_latex_cell(str(cell))}"
                    if not pd.isna(cell)
                    else " & -"
                )
            latex_lines.append(row_str + r" \\")

        latex_lines.extend(
            [
                r"\bottomrule",
                r"\end{tabular}",
                "}",
                f"\\label{{tab:{stage.label_prefix}{metric}}}",
                r"\end{table}",
                "",
            ]
        )
        with open(os.path.join(tables_dir, f"{metric}.tex"), "w") as f:
            f.write("\n".join(latex_lines))

    metrics = [
        "ari",
        "time_predict_ms",
        "time_fit_ms",
        "memory_bytes",
        "microcluster_count",
    ]

    int_path = os.path.join(DIR_METRICS, "aggregated_by_batch.parquet")
    if os.path.exists(int_path):
        df_int = pd.read_parquet(int_path)
        for dataset in stage.datasets:
            df_ds = df_int[df_int["dataset"] == dataset]
            fig, axes = plt.subplots(2, 3, figsize=(15, 8))
            for i, metric in enumerate(metrics):
                ax = axes.flatten()[i]
                for model in stage.models:
                    df_m = df_ds[df_ds["model"] == model].sort_values("batch")
                    if not df_m.empty:
                        header_label = IMPROVED_MODEL_SHORT_HEADERS.get(model, model)
                        clean_label = (
                            header_label.replace(r"\modelname{", "")
                            .replace("}", "")
                            .strip()
                        )
                        ax.plot(
                            df_m["batch"],
                            df_m[f"{metric}_median"],
                            label=clean_label,
                            color=IMPROVED_MODEL_COLORS.get(model, "#000000"),
                        )
                        ax.fill_between(
                            df_m["batch"],
                            df_m[f"{metric}_q1"],
                            df_m[f"{metric}_q3"],
                            color=IMPROVED_MODEL_COLORS.get(model, "#000000"),
                            alpha=0.15,
                        )
                ax.set_title(metric.replace("_", " ").title())
                ax.legend()
            axes.flatten()[-1].axis("off")
            plt.tight_layout()
            plt.savefig(
                os.path.join(plots_dir, f"{dataset}_trajectory.png"),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Run statistical analysis and artifact generation for improved models (Chapter 4)"
    )
    parser.add_argument(
        "--stage",
        type=str,
        default=None,
        help="Stage to evaluate (e.g. base_improved)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all defined improvement stages",
    )
    parser.add_argument(
        "--recompute-metrics",
        action="store_true",
        help="Force recomputation of parquet metrics from raw results CSVs",
    )
    args = parser.parse_args()

    if args.stage:
        if args.stage not in STAGE_BY_ID:
            raise ValueError(
                f"Unknown stage: {args.stage}. Valid stages: {list(STAGE_BY_ID.keys())}"
            )
        stages_to_run = [STAGE_BY_ID[args.stage]]
    else:
        stages_to_run = IMPROVEMENT_STAGES

    parquet_path = os.path.join(DIR_METRICS, "run_level.parquet")
    needs_recompute = False

    if args.recompute_metrics or not os.path.exists(parquet_path):
        needs_recompute = True
    else:
        parquet_mtime = os.path.getmtime(parquet_path)
        csv_files = glob.glob(os.path.join(DIR_RESULTS, "*.csv"))
        if any(os.path.getmtime(f) > parquet_mtime for f in csv_files):
            needs_recompute = True
        else:
            try:
                df_cached = pd.read_parquet(parquet_path, columns=["model"])
                if len(df_cached) != len(csv_files):
                    needs_recompute = True
                else:
                    cached_models = set(df_cached["model"].unique())
                    for stage in stages_to_run:
                        for m in stage.models:
                            if m not in cached_models:
                                if glob.glob(os.path.join(DIR_RESULTS, f"*_{m}_*.csv")):
                                    needs_recompute = True
                                    break
            except Exception:
                needs_recompute = True

    if needs_recompute:
        print(
            "Raw experiment CSVs changed or metrics out-of-date. Ingesting and recomputing Parquet metrics..."
        )
        compute_metrics()
        print("Parquet metrics generated.")

    for stage in stages_to_run:
        print(
            f"\nProcessing Improvement Stage: {stage.stage_id} ({stage.artifact_dir})"
        )
        print(f"  Models: {stage.models}")
        print(f"  Datasets: {len(stage.datasets)}")
        run_statistical_analysis(stage)
        generate_artifacts(stage)
        print(f"  Artifacts generated in {stage.artifact_dir}")

    print("\nAll requested improvement stages processed successfully.")


if __name__ == "__main__":
    main()
