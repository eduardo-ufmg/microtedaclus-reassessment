"""Statistical analysis: discretization per metric, paired test, effect size, and the
comparisons of a model in focus with each other model and of a variant with its control.
"""

import warnings

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon
from statsmodels.stats.multitest import multipletests

ALPHA = 0.05
CLIFFS_DELTA_THRESHOLD = 0.147

METRICS = {
    "ari_overall": {
        "dir": "max",
        "round": lambda x: np.round(x, 3),
        "fmt": "{:.3f}",
    },
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
    "memory_retained_bytes": {
        "dir": "min",
        "round": lambda x: np.round(x / 1024) * 1024,
        "fmt": "{:.0f}",
    },
    "memory_peak_bytes": {
        "dir": "min",
        "round": lambda x: np.round(x / 1024) * 1024,
        "fmt": "{:.0f}",
    },
    "mc_max": {"dir": "min", "round": lambda x: np.round(x), "fmt": "{:.0f}"},
    "mc_steadystate": {"dir": "min", "round": lambda x: np.round(x), "fmt": "{:.0f}"},
}


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    mat = np.sign(x[:, None] - y[None, :])
    return float(np.mean(mat))


def safe_wilcoxon(x: np.ndarray, y: np.ndarray) -> float:
    diff = x - y
    if np.all(diff == 0):
        return 1.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return float(wilcoxon(x, y).pvalue)
        except ValueError:
            return 1.0


def safe_mannwhitney(x: np.ndarray, y: np.ndarray) -> float:
    """Two-sided Mann-Whitney U test of two independent samples; 1 if they are constant and equal."""
    if np.all(x == x[0]) and np.all(y == x[0]):
        return 1.0
    return float(mannwhitneyu(x, y, alternative="two-sided").pvalue)


def compare_with_focus(
    df: pd.DataFrame, metric: str, focus: str, others: list
) -> pd.DataFrame:
    """The model in focus compared with each other model, in each dataset.

    Over the runs that every model of the comparison finished, each other model
    is compared with the model in focus by the paired test of
    `paired_comparisons`, and the p-values of a dataset are corrected together.
    The verdict of a row says how the model in focus compares with the model of
    that row; the row of the model in focus has none.
    """
    rows = []
    for dataset, of_dataset in df.groupby("dataset", sort=False):
        table = of_dataset.pivot(index="run", columns="model", values=metric)
        present = [model for model in others if model in table.columns]
        if focus not in table.columns or not present:
            continue
        table = table[[focus, *present]].dropna()
        if table.empty:
            continue
        pairs = {
            model: (table[model].to_numpy(), table[focus].to_numpy())
            for model in present
        }
        verdicts = paired_comparisons(pairs, metric)
        common = {"dataset": dataset, "runs": len(table)}
        rows.append(
            {
                **common,
                "model": focus,
                "median": verdicts["median"].iloc[0],
                "verdict": None,
            }
        )
        for model in present:
            rows.append(
                {
                    **common,
                    "model": model,
                    "median": verdicts.loc[model, "control_median"],
                    "verdict": verdicts.loc[model, "verdict"],
                }
            )
    return pd.DataFrame(rows)


def paired_comparisons(pairs: dict, metric: str) -> pd.DataFrame:
    """Paired comparison of a variant with its control in each group of observations.

    `pairs` maps a group (a dataset, a dimension) to the observations of the
    control and of the variant over the same runs. The p-values of the groups
    are corrected together.
    """
    config = METRICS[metric]
    rows = []
    for group, (control_values, variant_values) in pairs.items():
        x = np.asarray(config["round"](np.asarray(control_values, dtype=float)))
        y = np.asarray(config["round"](np.asarray(variant_values, dtype=float)))
        rows.append(
            {
                "group": group,
                "n": len(x),
                "control_median": float(np.median(x)),
                "control_q1": float(np.quantile(x, 0.25)),
                "control_q3": float(np.quantile(x, 0.75)),
                "median": float(np.median(y)),
                "q1": float(np.quantile(y, 0.25)),
                "q3": float(np.quantile(y, 0.75)),
                "p": safe_wilcoxon(y, x),
                "delta": cliffs_delta(y, x),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["p_adj"] = multipletests(out["p"], alpha=ALPHA, method="holm")[1]
    significant = (out["p_adj"] < ALPHA) & (
        out["delta"].abs() >= CLIFFS_DELTA_THRESHOLD
    )
    improves = out["delta"] > 0 if config["dir"] == "max" else out["delta"] < 0
    out["verdict"] = np.where(
        significant, np.where(improves, "better", "worse"), "same"
    )
    return out.set_index("group")
