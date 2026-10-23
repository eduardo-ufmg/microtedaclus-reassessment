"""
analyze_preprocessing_results.py

Preprocessing-impact experiment — ANALYSIS stage.

Loads the per-run results persisted by run_preprocessing_experiment.py from
./results/ (one CSV of per-batch curves + one params JSON per (stage, run))
and, for each stage, produces one metric figure:

    {stage}_metrics.png   median line + IQR (Q1-Q3) band

Tuned-hyperparameter summaries (median [Q1,Q3] for numeric, mode for
categorical) are shown as text on the figure. It also writes the figure of the
thesis, every stage in one figure, in the order of the preprocessing:

    monografia/MethodologyReassessment/artifacts/preprocessing/stages.png
"""

import glob
import json
import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pfc_shared.figures import FOCUS_COLOR

RESULTS_DIR = "./results"
PLOTS_DIR = "./plots/preprocessing_study"
os.makedirs(PLOTS_DIR, exist_ok=True)

STAGES = ["raw", "noise_removed", "downsampled", "ordered", "scaled"]
# the stages in the order in which the preprocessing applies them, with their titles
PIPELINE = [
    ("raw", "Raw"),
    ("noise_removed", "Noise removed"),
    ("downsampled", "Downsampled"),
    ("scaled", "Scaled"),
    ("ordered", "Ordered by label"),
]
THESIS_FIGURE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "..",
    "monografia",
    "MethodologyReassessment",
    "artifacts",
    "preprocessing",
    "stages.png",
)


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def discover_runs(stage):
    """Returns a sorted list of run indices for which both the curve CSV
    and the params JSON exist in RESULTS_DIR."""
    csv_paths = sorted(glob.glob(os.path.join(RESULTS_DIR, f"{stage}_run*.csv")))
    run_idxs = []
    for p in csv_paths:
        base = os.path.basename(p)[: -len(".csv")]  # "{stage}_run{idx:03d}"
        run_idx = int(base.split("_run")[-1])
        params_path = os.path.join(RESULTS_DIR, f"{stage}_run{run_idx:03d}_params.json")
        if os.path.exists(params_path):
            run_idxs.append(run_idx)
    return sorted(run_idxs)


def load_run(stage, run_idx):
    csv_path = os.path.join(RESULTS_DIR, f"{stage}_run{run_idx:03d}.csv")
    params_path = os.path.join(RESULTS_DIR, f"{stage}_run{run_idx:03d}_params.json")

    curve_df = pd.read_csv(csv_path)
    with open(params_path) as f:
        params = json.load(f)

    return (
        params,
        curve_df["ari"].to_numpy(),
        curve_df["t_predict_ms"].to_numpy(),
        curve_df["t_fit_ms"].to_numpy(),
    )


def load_stage(stage):
    """Loads all persisted runs for a stage. Arrays are truncated to the
    shortest run length (should normally all match, since eval-set size is
    fixed per stage; truncation only guards against partially-written or
    inconsistent runs)."""
    run_idxs = discover_runs(stage)
    if not run_idxs:
        return [], [], [], []

    params_runs, ari_runs, tp_runs, tf_runs = [], [], [], []
    for run_idx in run_idxs:
        params, ari, tp, tf = load_run(stage, run_idx)
        params_runs.append(params)
        ari_runs.append(ari)
        tp_runs.append(tp)
        tf_runs.append(tf)

    min_len = min(len(a) for a in ari_runs)
    ari_runs = [a[:min_len] for a in ari_runs]
    tp_runs = [a[:min_len] for a in tp_runs]
    tf_runs = [a[:min_len] for a in tf_runs]

    return params_runs, ari_runs, tp_runs, tf_runs


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------
def aggregate_curves_medianiqr(arr_list):
    """arr_list: list of 1D arrays of equal length (may contain NaNs for
    failed runs). Returns (median, q1, q3) along the run axis, ignoring
    NaNs."""
    stacked = np.vstack(arr_list)
    with np.errstate(all="ignore"):
        median = np.nanmedian(stacked, axis=0)
        q1 = np.nanpercentile(stacked, 25, axis=0)
        q3 = np.nanpercentile(stacked, 75, axis=0)
    return median, q1, q3


def aggregate_params(params_runs):
    """params_runs: list of dicts, one per run ({} for an all-pruned run).
    Numeric parameters are summarized by median [Q1, Q3] across valid runs;
    non-numeric (categorical) parameters by their mode. Returns
    (summary_dict[name -> formatted str], n_valid_runs)."""
    valid = [p for p in params_runs if p]
    if not valid:
        return {}, 0

    keys = valid[0].keys()
    summary = {}
    for k in keys:
        vals = [p[k] for p in valid if k in p]
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals):
            med = np.median(vals)
            q1 = np.percentile(vals, 25)
            q3 = np.percentile(vals, 75)

            if all(v > 1 for v in vals):
                summary[k] = f"{int(med)} [{int(q1)}, {int(q3)}]"
            else:
                summary[k] = f"{med:.1g} [{q1:.1g}, {q3:.1g}]"
        else:
            top, cnt = Counter(vals).most_common(1)[0]
            summary[k] = f"{top} ({cnt}/{len(vals)} runs)"
    return summary, len(valid)


def format_params_text(param_summary, n_valid, n_runs):
    if n_valid == 0:
        return "no valid trial in any run (all pruned)"
    lines = [f"{k} = {v}" for k, v in param_summary.items()]
    if n_valid < n_runs:
        lines.append(f"[{n_valid}/{n_runs} runs valid]")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------
def _plot_metrics(
    ari_stats,
    t_predict_stats,
    t_fit_stats,
    params_text,
    stage,
    out_dir,
):
    """*_stats: (median, q1, q3) tuples."""
    ari_c, ari_lo, ari_hi = ari_stats
    tp_c, tp_lo, tp_hi = t_predict_stats
    tf_c, tf_lo, tf_hi = t_fit_stats
    batches = np.arange(len(ari_c))

    center_label = "median"
    band_label = "IQR"

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(batches, ari_c, color="tab:blue", linewidth=2.0, label=center_label)
    axes[0].fill_between(
        batches, ari_lo, ari_hi, color="tab:blue", alpha=0.3, label=band_label
    )
    axes[0].set_title(f"{stage} — ARI per Batch)")
    axes[0].set_xlabel("Batch Index")
    axes[0].set_ylabel("Adjusted Rand Index")
    axes[0].set_ylim(-1.0, 1.0)
    axes[0].legend(fontsize=7, loc="upper right")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    axes[0].text(
        0.02,
        0.03,
        params_text,
        transform=axes[0].transAxes,
        ha="left",
        va="bottom",
        bbox=dict(
            boxstyle="round,pad=0.35", facecolor="white", edgecolor="gray", alpha=0.85
        ),
    )

    axes[1].plot(
        batches,
        tp_c,
        label=f"predict ({center_label})",
        color="tab:orange",
        linewidth=2.0,
    )
    axes[1].fill_between(
        batches,
        tp_lo,
        tp_hi,
        color="tab:orange",
        alpha=0.3,
        label=f"predict {band_label}",
    )
    axes[1].plot(
        batches, tf_c, label=f"fit ({center_label})", color="tab:green", linewidth=2.0
    )
    axes[1].fill_between(
        batches, tf_lo, tf_hi, color="tab:green", alpha=0.3, label=f"fit {band_label}"
    )
    axes[1].set_title(
        f"{stage} — Processing Time per Batch ({center_label}/{band_label})"
    )
    axes[1].set_xlabel("Batch Index")
    axes[1].set_ylabel("Time (ms)")
    axes[1].set_yscale("log")
    axes[1].legend(fontsize=7)
    axes[1].grid(True, which="both", linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{stage}_metrics.png"))
    plt.close(fig)


def plot_metrics(ari_runs, tp_runs, tf_runs, params_text, stage, out_dir):
    _plot_metrics(
        aggregate_curves_medianiqr(ari_runs),
        aggregate_curves_medianiqr(tp_runs),
        aggregate_curves_medianiqr(tf_runs),
        params_text,
        stage,
        out_dir,
    )


def plot_pipeline(path=THESIS_FIGURE):
    """Every stage in one figure: its ARI per batch, with the hyperparameters selected
    for it, and its learning and prediction times per batch; medians and
    interquartile bands over the runs."""
    fig, axes = plt.subplots(len(PIPELINE), 2, figsize=(12, 15))
    for (ax_ari, ax_time), (stage, title) in zip(axes, PIPELINE):
        params_runs, ari_runs, tp_runs, tf_runs = load_stage(stage)
        summary, n_valid = aggregate_params(params_runs)
        batches = np.arange(len(ari_runs[0]))

        median, q1, q3 = aggregate_curves_medianiqr(ari_runs)
        ax_ari.plot(batches, median, color=FOCUS_COLOR, linewidth=1.8)
        ax_ari.fill_between(batches, q1, q3, color=FOCUS_COLOR, alpha=0.25, linewidth=0)
        ax_ari.set_ylim(-0.1, 1.05)
        ax_ari.set_title(f"{title}: ARI", fontsize=10)
        ax_ari.set_ylabel("ARI")
        ax_ari.text(
            0.98,
            0.95,
            format_params_text(summary, n_valid, len(params_runs)),
            transform=ax_ari.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray"),
        )

        for runs, label, style in (
            (tf_runs, "learning", "-"),
            (tp_runs, "prediction", "--"),
        ):
            median, q1, q3 = aggregate_curves_medianiqr(runs)
            ax_time.plot(
                batches,
                median,
                color=FOCUS_COLOR,
                linestyle=style,
                linewidth=1.5,
                label=label,
            )
            ax_time.fill_between(
                batches, q1, q3, color=FOCUS_COLOR, alpha=0.15, linewidth=0
            )
        ax_time.set_yscale("log")
        ax_time.set_title(f"{title}: time per batch [ms]", fontsize=10)
        ax_time.legend(fontsize=8, loc="upper right")
    for ax in axes[-1]:
        ax.set_xlabel("Batch")
    plt.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    for stage in STAGES:
        params_runs, ari_runs, tp_runs, tf_runs = load_stage(stage)

        if not ari_runs:
            print(f"[{stage}] No persisted runs found in {RESULTS_DIR}/ — skipping.")
            continue

        param_summary, n_valid = aggregate_params(params_runs)
        params_text = format_params_text(param_summary, n_valid, len(params_runs))

        if n_valid == 0:
            print(f"[{stage}] All trials pruned in every run — skipping plots.")
            continue

        plot_metrics(ari_runs, tp_runs, tf_runs, params_text, stage, PLOTS_DIR)

        print(
            f"[{stage}] {len(ari_runs)} runs loaded ({n_valid} valid) — plots written."
        )

    plot_pipeline()
    print(f"Done. Analysis plots written to {PLOTS_DIR}/ and {THESIS_FIGURE}")


if __name__ == "__main__":
    main()
