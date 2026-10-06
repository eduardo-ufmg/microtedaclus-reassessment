#!/usr/bin/env python3
"""Unified statistical analysis and LaTeX artifact generation pipeline."""

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
from stages_config import (
    ALL_STAGES,
    MODEL_COLORS,
    MODEL_SHORT_HEADERS,
    READABLE_METRICS,
    STAGE_BY_ID,
    STATS_METRICS,
    Stage,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPOSITORY_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))

ALPHA = 0.05
CLIFFS_DELTA_THRESHOLD = 0.147


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
                    return None
    return None


def compute_metrics(results_dir: str, metrics_dir: str):
    """Ingest raw execution CSVs and generate parquet metric caches."""
    os.makedirs(metrics_dir, exist_ok=True)
    csv_files = glob.glob(os.path.join(results_dir, "*.csv"))
    if not csv_files:
        print(f"Warning: No CSV files found in {results_dir}")
        return

    records_run = []
    records_batch = []

    print(f"Ingesting {len(csv_files)} raw CSVs...")
    for f in csv_files:
        parsed = parse_result_filename(f)
        if not parsed:
            continue
        dataset, model, run = parsed

        try:
            df = pd.read_csv(f)
            if df.empty or "batch" not in df.columns:
                continue

            # Ensure numeric conversion
            numeric_cols = [
                "ari",
                "time_predict_cpu_ns",
                "time_fit_cpu_ns",
                "memory_bytes",
                "microcluster_count",
            ]
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

            # Scalar Metrics
            ari_overall = df["ari"].mean()
            ari_fifth = df["ari"].iloc[4] if len(df) > 4 else df["ari"].iloc[-1]
            ss_start = int(len(df) * 0.8)
            ari_steadystate = df["ari"].iloc[ss_start:].mean()
            ari_diff = max(0.0, df["ari"].max() - df["ari"].min())

            t_pred_cum_ms = df["time_predict_cpu_ns"].sum() / 1e6
            t_fit_cum_ms = df["time_fit_cpu_ns"].sum() / 1e6

            mem_max = df["memory_bytes"].max()
            mem_ss = df["memory_bytes"].iloc[ss_start:].mean()

            mc_max = df["microcluster_count"].max()
            mc_ss = df["microcluster_count"].iloc[ss_start:].mean()

            records_run.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "run": run,
                    "ari_overall": ari_overall,
                    "ari_fifth": ari_fifth,
                    "ari_steadystate": ari_steadystate,
                    "ari_diff": ari_diff,
                    "time_predict_cum_ms": t_pred_cum_ms,
                    "time_fit_cum_ms": t_fit_cum_ms,
                    "memory_max_bytes": mem_max,
                    "memory_steadystate_bytes": mem_ss,
                    "mc_max": mc_max,
                    "mc_steadystate": mc_ss,
                }
            )

            # Batch Trajectory Metrics
            df_batch = df[
                [
                    "batch",
                    "ari",
                    "time_predict_cpu_ns",
                    "time_fit_cpu_ns",
                    "memory_bytes",
                    "microcluster_count",
                ]
            ].copy()
            df_batch["time_predict_ms"] = df_batch["time_predict_cpu_ns"] / 1e6
            df_batch["time_fit_ms"] = df_batch["time_fit_cpu_ns"] / 1e6
            df_batch["dataset"] = dataset
            df_batch["model"] = model
            df_batch["run"] = run
            records_batch.append(df_batch)

        except Exception as e:
            continue

    if records_run:
        df_runs = pd.DataFrame(records_run)
        df_runs.to_parquet(os.path.join(metrics_dir, "run_level.parquet"), index=False)

    if records_batch:
        df_batches = pd.concat(records_batch, ignore_index=True)
        # Aggregated quantiles by batch
        grouped = df_batches.groupby(["dataset", "model", "batch"])
        agg_funcs = {}
        for col in [
            "ari",
            "time_predict_ms",
            "time_fit_ms",
            "memory_bytes",
            "microcluster_count",
        ]:
            agg_funcs[f"{col}_median"] = (col, "median")
            agg_funcs[f"{col}_q1"] = (col, lambda x: x.quantile(0.25))
            agg_funcs[f"{col}_q3"] = (col, lambda x: x.quantile(0.75))
        df_agg_batch = grouped.agg(**agg_funcs).reset_index()
        df_agg_batch.to_parquet(
            os.path.join(metrics_dir, "aggregated_by_batch.parquet"), index=False
        )


def run_statistical_analysis(stage: Stage, metrics_dir: str, stats_dir: str):
    """Run non-parametric statistical evaluation (Friedman, Wilcoxon, Cliff's delta)."""
    parquet_path = os.path.join(metrics_dir, "run_level.parquet")
    if not os.path.exists(parquet_path):
        return

    df_runs = pd.read_parquet(parquet_path)
    stage_stats_dir = os.path.join(stats_dir, stage.stage_id)
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
            if df_ds.empty:
                for m in stage.models:
                    row_dict[m] = "-"
                results.append(row_dict)
                continue

            # Continuous statistics
            medians = df_ds.median()
            q1s = df_ds.quantile(0.25)
            q3s = df_ds.quantile(0.75)

            # Discretized values for ties/effect size
            df_disc = pd.DataFrame(
                config["round"](df_ds.values), columns=valid_models, index=df_ds.index
            )
            medians_disc = df_disc.median()

            best_tier = set(valid_models)
            worst_tier = set(valid_models)

            # Run significance tests only if >= 2 paired samples exist
            if len(df_ds) >= 2:
                data_by_model = [df_ds[m].values for m in valid_models]
                friedman_p = 1.0
                if not all(np.all(d == data_by_model[0]) for d in data_by_model):
                    try:
                        _, friedman_p = friedmanchisquare(*data_by_model)
                    except Exception:
                        friedman_p = 1.0
                is_max = config["dir"] == "max"
                sorted_models = sorted(
                    valid_models,
                    key=lambda m: (medians_disc[m], medians[m]),
                    reverse=is_max,
                )
                best_model = sorted_models[0]
                worst_model = sorted_models[-1]

                def process_tier(ref_model, tier_set):
                    p_vals = []
                    comps = []
                    for m in valid_models:
                        if m == ref_model:
                            continue
                        diff = df_disc[ref_model] - df_disc[m]
                        if np.all(diff == 0):
                            continue
                        try:
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                _, p = wilcoxon(df_ds[ref_model], df_ds[m])
                            p_vals.append(p)
                            comps.append(m)
                        except Exception:
                            pass

                    if p_vals:
                        rej, _, _, _ = multipletests(p_vals, alpha=ALPHA, method="holm")
                        for m, r in zip(comps, rej):
                            if r:
                                n1 = len(df_disc[ref_model])
                                n2 = len(df_disc[m])
                                d_val = (
                                    np.sum(
                                        df_disc[ref_model].values[:, None]
                                        > df_disc[m].values
                                    )
                                    - np.sum(
                                        df_disc[ref_model].values[:, None]
                                        < df_disc[m].values
                                    )
                                ) / (n1 * n2)
                                if is_max:
                                    if (
                                        ref_model == best_model
                                        and d_val >= CLIFFS_DELTA_THRESHOLD
                                    ):
                                        tier_set.discard(m)
                                    elif (
                                        ref_model == worst_model
                                        and d_val <= -CLIFFS_DELTA_THRESHOLD
                                    ):
                                        tier_set.discard(m)
                                else:
                                    if (
                                        ref_model == best_model
                                        and d_val <= -CLIFFS_DELTA_THRESHOLD
                                    ):
                                        tier_set.discard(m)
                                    elif (
                                        ref_model == worst_model
                                        and d_val >= CLIFFS_DELTA_THRESHOLD
                                    ):
                                        tier_set.discard(m)

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

            for m in stage.models:
                if m not in row_dict:
                    row_dict[m] = "-"

            results.append(row_dict)

        if results:
            pd.DataFrame(results).set_index("dataset").to_csv(
                os.path.join(stage_stats_dir, f"{metric}_statistical_analysis.csv")
            )


def format_latex_cell(stat_cell: str) -> str:
    """Format table cell with bold/red highlights based on statistical tiers."""
    if str(stat_cell) in ("nan", "-"):
        return "-"
    parts = str(stat_cell).split(" [")
    base = parts[0]
    tags = parts[1].replace("]", "") if len(parts) > 1 else ""
    if "B" in tags and "W" in tags:
        return base
    if "B" in tags:
        return f"\\textbf{{{base}}}"
    if "W" in tags:
        return f"\\textcolor{{red}}{{{base}}}"
    return base


def generate_artifacts(stage: Stage, stats_dir: str, metrics_dir: str):
    """Generate publication-ready LaTeX tables and trajectory plots."""
    stage_stats_dir = os.path.join(stats_dir, stage.stage_id)
    out_chapter_dir = os.path.join(
        REPOSITORY_ROOT, "monografia", stage.chapter, stage.artifact_dir
    )
    tables_dir = os.path.join(out_chapter_dir, "tables")
    plots_dir = os.path.join(out_chapter_dir, "plots")
    os.makedirs(stage_stats_dir, exist_ok=True)
    os.makedirs(out_chapter_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    stat_files = [f for f in os.listdir(stage_stats_dir) if f.endswith(".csv")]
    for stat_file in stat_files:
        metric = stat_file.replace("_statistical_analysis.csv", "")
        df_stat = pd.read_csv(
            os.path.join(stage_stats_dir, stat_file), index_col="dataset"
        )
        ds_order = {d: i for i, d in enumerate(stage.datasets)}
        df_stat = df_stat.sort_index(
            key=lambda idx: idx.map(lambda d: ds_order.get(d, len(stage.datasets)))
        )

        headers = [
            MODEL_SHORT_HEADERS.get(m, f"\\modelname{{{m}}}") for m in stage.models
        ]

        latex_lines = [
            "\\begin{table}[htbp]",
            "\\centering",
            f"\\caption{{Statistical analysis for the {READABLE_METRICS.get(metric, metric.replace('_', ' ').title())}.}}",
            "\\resizebox{\\textwidth}{!}{%",
            "\\begin{tabular}{l " + "c " * len(stage.models) + "}",
            "\\toprule",
            "Dataset & " + " & ".join(headers) + " \\\\",
            "\\midrule",
        ]

        for dataset in df_stat.index:
            display_name = dataset_display_name(dataset).replace("_", "\\_")
            row_str = f"\\datasetname{{{display_name}}}"
            for model in stage.models:
                cell = df_stat.loc[dataset, model] if model in df_stat.columns else "-"
                row_str += f" & {format_latex_cell(str(cell))}"
            latex_lines.append(row_str + " \\\\")

        latex_lines.extend(
            [
                "\\bottomrule",
                "\\end{tabular}",
                "}",
                f"\\label{{tab:{stage.label_prefix}{metric}}}",
                "\\end{table}",
            ]
        )
        with open(os.path.join(tables_dir, f"{metric}.tex"), "w") as f:
            f.write("\n".join(latex_lines))

    # Trajectory plots
    int_path = os.path.join(metrics_dir, "aggregated_by_batch.parquet")
    if os.path.exists(int_path):
        df_int = pd.read_parquet(int_path)
        plot_metrics = [
            "ari",
            "time_predict_ms",
            "time_fit_ms",
            "memory_bytes",
            "microcluster_count",
        ]
        for dataset in df_int["dataset"].unique():
            if dataset not in stage.datasets:
                continue
            df_ds = df_int[df_int["dataset"] == dataset]
            fig, axes = plt.subplots(2, 3, figsize=(15, 8))
            for i, metric in enumerate(plot_metrics):
                ax = axes.flatten()[i]
                for model in stage.models:
                    df_m = df_ds[df_ds["model"] == model].sort_values("batch")
                    if not df_m.empty:
                        ax.plot(
                            df_m["batch"],
                            df_m[f"{metric}_median"],
                            label=MODEL_SHORT_HEADERS.get(model, model),
                            color=MODEL_COLORS.get(model, "#000000"),
                            linewidth=1.5,
                        )
                        ax.fill_between(
                            df_m["batch"],
                            df_m[f"{metric}_q1"],
                            df_m[f"{metric}_q3"],
                            color=MODEL_COLORS.get(model, "#000000"),
                            alpha=0.15,
                        )
                ax.set_title(metric.replace("_", " ").title())
                ax.set_xlabel("Batch Index")
                if metric != "ari":
                    ax.set_yscale("log")
                if i == 0:
                    ax.legend(loc="best", fontsize=8)
            axes.flatten()[-1].axis("off")
            plt.suptitle(f"{dataset_display_name(dataset)} ({stage.stage_id})", y=1.02)
            plt.tight_layout()
            fig.savefig(
                os.path.join(plots_dir, f"{dataset}_trajectory.png"),
                dpi=150,
                bbox_inches="tight",
            )
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Unified streaming clustering analysis and LaTeX table generation"
    )
    parser.add_argument(
        "--stage",
        type=str,
        default=None,
        help="Specific stage ID to process (e.g. order_dependency, base_improved)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all defined stages",
    )
    parser.add_argument(
        "--recompute-metrics",
        action="store_true",
        help="Force recomputation of parquet metrics from raw results CSVs",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="./results_experiments",
        help="Directory containing raw batch CSVs (default: ./results_experiments)",
    )
    parser.add_argument(
        "--metrics-dir",
        type=str,
        default="./metrics_experiments",
        help="Directory to save/load parquet metrics (default: ./metrics_experiments)",
    )
    parser.add_argument(
        "--stats-dir",
        type=str,
        default="./statistical_analysis_experiments",
        help="Directory to save statistical CSVs (default: ./statistical_analysis_experiments)",
    )
    args = parser.parse_args()

    if args.stage:
        if args.stage not in STAGE_BY_ID:
            raise ValueError(
                f"Unknown stage '{args.stage}'. Valid stages: {list(STAGE_BY_ID.keys())}"
            )
        stages_to_run = [STAGE_BY_ID[args.stage]]
    elif args.all:
        stages_to_run = ALL_STAGES
    else:
        stages_to_run = ALL_STAGES

    parquet_path = os.path.join(args.metrics_dir, "run_level.parquet")
    needs_recompute = args.recompute_metrics or not os.path.exists(parquet_path)

    if not needs_recompute and os.path.exists(parquet_path):
        parquet_mtime = os.path.getmtime(parquet_path)
        csv_files = glob.glob(os.path.join(args.results_dir, "*.csv"))
        if any(os.path.getmtime(f) > parquet_mtime for f in csv_files):
            needs_recompute = True

    if needs_recompute:
        print(f"Ingesting raw CSVs from {args.results_dir} into {args.metrics_dir}...")
        compute_metrics(args.results_dir, args.metrics_dir)

    for stage in stages_to_run:
        print(
            f"\nProcessing Stage: {stage.stage_id} ({stage.chapter} / {stage.artifact_dir})"
        )
        print(f"  Models: {stage.models}")
        print(f"  Datasets: {len(stage.datasets)}")
        run_statistical_analysis(stage, args.metrics_dir, args.stats_dir)
        generate_artifacts(stage, args.stats_dir, args.metrics_dir)
        print(
            f"  Artifacts generated in monografia/{stage.chapter}/{stage.artifact_dir}/"
        )

    print("\nAll requested stages processed successfully.")


if __name__ == "__main__":
    main()
