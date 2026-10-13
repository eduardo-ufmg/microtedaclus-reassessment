"""Statistical analysis: discretization per metric, paired test, effect size and the two comparisons."""

import warnings

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, mannwhitneyu, wilcoxon
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


def process_metric(
    df: pd.DataFrame, metric: str, config: dict, subset_models: list
) -> pd.DataFrame:
    results = []
    datasets = df["dataset"].unique()

    for dataset in datasets:
        df_ds = df[df["dataset"] == dataset].pivot(
            index="run", columns="model", values=metric
        )

        valid_models = [m for m in subset_models if m in df_ds.columns]
        if len(valid_models) < 2:
            continue

        df_ds = df_ds[valid_models].dropna()
        if df_ds.empty:
            continue

        data_matrix = config["round"](df_ds.values)
        df_disc = pd.DataFrame(data_matrix, columns=valid_models, index=df_ds.index)

        medians = df_disc.median()
        q1s = df_disc.quantile(0.25)
        q3s = df_disc.quantile(0.75)

        if config["dir"] == "max":
            best_model = medians.idxmax()
            worst_model = medians.idxmin()
        else:
            best_model = medians.idxmin()
            worst_model = medians.idxmax()

        # The Friedman test needs at least three models. With two, the pairwise
        # test below is the whole comparison, so no omnibus test precedes it.
        is_omnibus_sig = len(valid_models) == 2
        if not is_omnibus_sig:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    stat, p_friedman = friedmanchisquare(
                        *[df_disc[m].values for m in valid_models]
                    )
                    if p_friedman < ALPHA:
                        is_omnibus_sig = True
                except ValueError:
                    pass

        best_tier = set([best_model])
        worst_tier = set([worst_model])

        if not is_omnibus_sig:
            best_tier = set(valid_models)
            worst_tier = set(valid_models)
        else:
            pvals_best = []
            models_to_test_best = [m for m in valid_models if m != best_model]
            for m in models_to_test_best:
                pvals_best.append(
                    safe_wilcoxon(df_disc[best_model].to_numpy(), df_disc[m].values)
                )

            pvals_worst = []
            models_to_test_worst = [m for m in valid_models if m != worst_model]
            for m in models_to_test_worst:
                pvals_worst.append(
                    safe_wilcoxon(df_disc[worst_model].to_numpy(), df_disc[m].values)
                )

            _, pvals_best_adj, _, _ = multipletests(
                pvals_best, alpha=ALPHA, method="holm"
            )
            _, pvals_worst_adj, _, _ = multipletests(
                pvals_worst, alpha=ALPHA, method="holm"
            )

            for m, p_adj in zip(models_to_test_best, pvals_best_adj):
                d = cliffs_delta(df_disc[best_model].to_numpy(), df_disc[m].values)
                if p_adj >= ALPHA or abs(d) < CLIFFS_DELTA_THRESHOLD:
                    best_tier.add(m)

            for m, p_adj in zip(models_to_test_worst, pvals_worst_adj):
                d = cliffs_delta(df_disc[worst_model].to_numpy(), df_disc[m].values)
                if p_adj >= ALPHA or abs(d) < CLIFFS_DELTA_THRESHOLD:
                    worst_tier.add(m)

        row_dict = {"dataset": dataset}
        fmt = config["fmt"]
        for m in valid_models:
            med_val = medians[m]
            q1_val = q1s[m]
            q3_val = q3s[m]

            base_str = (
                f"{fmt.format(med_val)} ({fmt.format(q1_val)}, {fmt.format(q3_val)})"
            )

            tags = []
            if m in best_tier:
                tags.append("B")
            if m in worst_tier:
                tags.append("W")

            tag_str = f" [{','.join(tags)}]" if tags else ""
            row_dict[m] = f"{base_str}{tag_str}"

        results.append(row_dict)

    if not results:
        return pd.DataFrame()

    return pd.DataFrame(results).set_index("dataset")


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
