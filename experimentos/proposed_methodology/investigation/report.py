"""Run-level metrics and paired comparisons for investigation variants.

Metrics follow analysis_pipeline.py: Overall ARI (mean over batches), ARI at the
fifth batch, Steady-State ARI (mean over the last 20% of batches) and Maximum
ARI Drop (max - min). `official:<model>` loads results_experiments/.

Usage:
    python investigation/report.py REFERENCE VARIANT [VARIANT ...] [--metric ari_overall]
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from editorial_config import ITERATIVE_DATASET_ORDER  # noqa: E402

RESULTS_DIR = ROOT / "results_investigation"
QUALITY = ["ari_overall", "ari_fifth", "ari_steadystate", "ari_diff"]
FIRST_RUN = 0  # 32 selects the confirmation seeds
SHORT = {
    "benchmark_1": "B1",
    "benchmark_2": "B2",
    "random_rbf_events": "RBF",
    "cassini": "CSN",
    "chameleon_ds3": "DS3",
    "chameleon_ds4": "DS4",
    "SCL-32": "S32",
    "SCL-64": "S64",
}


def _file(variant, dataset, run):
    if variant.startswith("official:"):
        return (
            ROOT
            / "results_experiments"
            / f"{dataset}_{variant.split(':', 1)[1]}_{run}.csv"
        )
    return RESULTS_DIR / variant / f"{dataset}_{run}.csv"


def run_level(variant, runs=32, datasets=None, first_run=None):
    rows = []
    first_run = FIRST_RUN if first_run is None else first_run
    for dataset in datasets or ITERATIVE_DATASET_ORDER:
        for run in range(first_run, first_run + runs):
            f = _file(variant, dataset, run)
            if not f.exists():
                continue
            df = pd.read_csv(f)
            if df.empty or "ari" not in df.columns or len(df) < 2:
                continue
            ss = int(len(df) * 0.8)
            row = {
                "dataset": dataset,
                "run": run,
                "ari_overall": df["ari"].mean(),
                "ari_fifth": df["ari"].iloc[4] if len(df) > 4 else df["ari"].iloc[-1],
                "ari_steadystate": df["ari"].iloc[ss:].mean(),
                "ari_diff": max(0.0, df["ari"].max() - df["ari"].min()),
                "mc_steadystate": df["microcluster_count"].iloc[ss:].mean(),
                "time_fit_ms": df["time_fit_cpu_ns"].sum() / 1e6,
                "time_predict_ms": df["time_predict_cpu_ns"].sum() / 1e6,
            }
            for extra in ("n_pred", "homogeneity", "completeness", "noise_pred"):
                if extra in df.columns:
                    row[extra] = df[extra].mean()
            rows.append(row)
    return pd.DataFrame(rows)


def compare(reference, variant, metric="ari_overall", runs=32, alpha=0.05):
    """Paired per-dataset comparison of `variant` against `reference`."""
    ref, var = run_level(reference, runs), run_level(variant, runs)
    if ref.empty or var.empty:
        return pd.DataFrame()
    higher_is_better = metric != "ari_diff"
    rows, pvals = [], []
    for dataset in ITERATIVE_DATASET_ORDER:
        a = ref[ref.dataset == dataset].set_index("run")[metric]
        b = var[var.dataset == dataset].set_index("run")[metric]
        common = a.index.intersection(b.index)
        a, b = a.loc[common], b.loc[common]
        if len(common) == 0:
            continue
        diff = (b - a) if higher_is_better else (a - b)
        p = 1.0
        if len(common) >= 6 and not np.allclose(diff, 0):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                p = float(wilcoxon(b, a).pvalue)
        pvals.append(p)
        rows.append(
            {
                "dataset": SHORT[dataset],
                "n": len(common),
                "ref": a.median(),
                "var": b.median(),
                "wins": int((diff > 1e-9).sum()),
                "losses": int((diff < -1e-9).sum()),
                "sign": np.sign(np.median(diff)) or np.sign(diff.mean()),
                "p": p,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["p_holm"] = multipletests(out["p"], alpha=alpha, method="holm")[1]
    out["verdict"] = np.where(
        out.p_holm < alpha, np.where(out.sign > 0, "BETTER", "WORSE"), "="
    )
    return out


def table(variants, metric="ari_overall", runs=32):
    """Median of a run-level metric per dataset for several variants."""
    cols = {}
    for v in variants:
        rl = run_level(v, runs)
        if rl.empty:
            continue
        cols[v] = (
            rl.groupby("dataset")[metric].median().reindex(ITERATIVE_DATASET_ORDER)
        )
    df = pd.DataFrame(cols).T
    df.columns = [SHORT[c] for c in df.columns]
    df["mean"] = df.mean(axis=1)
    return df


def verdict_line(reference, variant, runs=32):
    parts = []
    for metric in QUALITY:
        c = compare(reference, variant, metric, runs)
        if c.empty:
            continue
        better = ",".join(c[c.verdict == "BETTER"].dataset)
        worse = ",".join(c[c.verdict == "WORSE"].dataset)
        parts.append(f"{metric[4:]}: +[{better}] -[{worse}]")
    return f"{variant} vs {reference} | " + " | ".join(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reference")
    parser.add_argument("variants", nargs="+")
    parser.add_argument("--runs", type=int, default=32)
    parser.add_argument("--metric", default=None)
    parser.add_argument("--detail", action="store_true")
    parser.add_argument("--first-run", type=int, default=0)
    args = parser.parse_args()
    global FIRST_RUN
    FIRST_RUN = args.first_run
    pd.set_option("display.width", 250)
    pd.set_option("display.float_format", lambda x: f"{x:.3f}")
    metrics = [args.metric] if args.metric else ["ari_overall", "ari_steadystate"]
    for metric in metrics:
        print(f"== median {metric}")
        print(table([args.reference, *args.variants], metric, args.runs))
    for v in args.variants:
        print(verdict_line(args.reference, v, args.runs))
        if args.detail:
            for metric in metrics:
                print(
                    compare(args.reference, v, metric, args.runs).to_string(index=False)
                )


if __name__ == "__main__":
    main()
