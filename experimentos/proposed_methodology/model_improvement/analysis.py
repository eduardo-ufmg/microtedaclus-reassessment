"""Run-level metrics and statistical comparisons of the model improvement experiments.

The metrics are computed by the code of the main experiment (compute_metrics.py)
and the tests use the ingredients of its statistical analysis (statistic_analysis.py):
the same discretization, the paired Wilcoxon signed-rank test, the Holm correction
and the Cliff's delta threshold. The only adaptation is the reference of the
comparisons: a variant is compared with its control, not with the best and worst
models of a table, so no omnibus test is involved. The p-values of the datasets
of one comparison are corrected together.
"""

from functools import lru_cache

import numpy as np
import pandas as pd

from compute_metrics import add_time_columns, is_valid_result, scalar_metrics
from model_improvement.experiments import DATASETS
from model_improvement.harness import result_paths
from pfc_shared.statistics import METRICS, paired_comparisons

# A change is accepted or rejected on these metrics only; the others are reported.
DECISION_METRICS = ("ari_overall", "ari_steadystate")
DIAGNOSTICS = ("n_true", "n_pred", "homogeneity", "completeness", "noise_pred")


@lru_cache(maxsize=None)
def run_level(variant: str, runs: range) -> pd.DataFrame:
    """One row per (dataset, run) with the ten metrics of the main experiment."""
    rows = []
    for dataset in DATASETS:
        for run in runs:
            out_file, par_file = result_paths(variant, dataset, run)
            if not out_file.exists():
                continue
            df = pd.read_csv(out_file)
            if "status" not in df.columns or not is_valid_result(df):
                continue
            row = {
                "dataset": dataset,
                "run": run,
                **scalar_metrics(add_time_columns(df)),
            }
            for name in DIAGNOSTICS:
                if name in df.columns:
                    row[name] = df[name].mean()
            if "n_pred" in df.columns:
                # The first batch is predicted before any sample is learned, so it is left out.
                predicted = df["n_pred"].to_numpy()
                row["single_cluster_share"] = float(np.mean(predicted[1:] == 1))
                row["ends_single_cluster"] = bool(predicted[-2] == 1)
            if par_file.exists():
                params = pd.read_csv(par_file)
                if not params.empty:
                    for name, value in zip(params["param_name"], params["param_value"]):
                        row[f"param:{name}"] = float(value)
            rows.append(row)
    return pd.DataFrame(rows)


def compare(control: str, variant: str, metric: str, runs: range) -> pd.DataFrame:
    """Per-dataset paired comparison of a variant with its control on one metric."""
    ctrl, var = run_level(control, runs), run_level(variant, runs)
    if ctrl.empty or var.empty:
        return pd.DataFrame()
    pairs = {}
    for dataset in DATASETS:
        a = ctrl[ctrl.dataset == dataset].set_index("run")[metric]
        b = var[var.dataset == dataset].set_index("run")[metric]
        common = a.index.intersection(b.index)
        if len(common):
            pairs[dataset] = (a.loc[common].to_numpy(), b.loc[common].to_numpy())
    return paired_comparisons(pairs, metric).rename_axis("dataset")


def decide(control: str, variant: str, runs: range) -> dict:
    """Decision rule: no dataset worse and at least one better, on the decision metrics."""
    better, worse = {}, {}
    for metric in DECISION_METRICS:
        c = compare(control, variant, metric, runs)
        if c.empty:
            return {"outcome": "missing", "better": {}, "worse": {}}
        better[metric] = list(c.index[c.verdict == "better"])
        worse[metric] = list(c.index[c.verdict == "worse"])
    any_better, any_worse = any(better.values()), any(worse.values())
    if any_better and not any_worse:
        outcome = "improves"
    elif any_worse and not any_better:
        outcome = "regresses"
    elif any_better and any_worse:
        outcome = "mixed"
    else:
        outcome = "indistinguishable"
    return {"outcome": outcome, "better": better, "worse": worse}


def medians(variant: str, metric: str, runs: range) -> pd.Series:
    """Median over the runs of the discretized metric, per dataset."""
    rl = run_level(variant, runs)
    if rl.empty:
        return pd.Series(dtype=float)
    disc = rl.assign(value=METRICS[metric]["round"](rl[metric].to_numpy(dtype=float)))
    return disc.groupby("dataset")["value"].median().reindex(DATASETS)
