#!/usr/bin/env python3
import os
import warnings
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon
from statsmodels.stats.multitest import multipletests

from editorial_config import SUBSETS

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DIR_METRICS_NON_SCL = os.path.join(SCRIPT_DIR, "metrics", "non_scalability")
DIR_OUT = os.path.join(SCRIPT_DIR, "statistical_analysis")

ALPHA = 0.05
CLIFFS_DELTA_THRESHOLD = 0.147

METRICS = {
    "ari_overall": {
        "dir": "max",
        "round": lambda x: np.round(x, 3),
        "fmt": "{:.3f}",
    },
    "ari_penultimate": {
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
    "memory_penultimate_bytes": {
        "dir": "min",
        "round": lambda x: np.round(x / 1024) * 1024,
        "fmt": "{:.0f}",
    },
    "mc_max": {"dir": "min", "round": lambda x: np.round(x), "fmt": "{:.0f}"},
    "mc_penultimate": {"dir": "min", "round": lambda x: np.round(x), "fmt": "{:.0f}"},
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

        is_omnibus_sig = False
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

    return pd.DataFrame(results).set_index("dataset")


def main():
    path_in = os.path.join(DIR_METRICS_NON_SCL, "run_level.parquet")
    if not os.path.exists(path_in):
        print(f"File not found: {path_in}")
        return

    df_runs = pd.read_parquet(path_in)

    for subset_name, subset_models in SUBSETS.items():
        subset_out_dir = os.path.join(DIR_OUT, subset_name)
        os.makedirs(subset_out_dir, exist_ok=True)

        for metric, config in METRICS.items():
            if metric not in df_runs.columns:
                continue

            df_table = process_metric(df_runs, metric, config, subset_models)
            out_path = os.path.join(
                subset_out_dir, f"{metric}_statistical_analysis.csv"
            )
            df_table.to_csv(out_path)
        print(f"Generated statistical relevance tables for subset: {subset_name}")


if __name__ == "__main__":
    main()
