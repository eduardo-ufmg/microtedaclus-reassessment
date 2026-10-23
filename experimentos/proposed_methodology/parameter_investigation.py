#!/usr/bin/env python3
"""Hyperparameters selected for the model in focus of each comparison.

One table per model, in the chapter of its comparison: each hyperparameter is a
row under its model, each dataset a column, and a cell gives the median and the
quartile coefficient of dispersion of the values selected over the runs.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

from editorial_config import (
    COMPARISONS,
    DATASET_ORDER,
    DATASET_SHORT_NAMES,
    model_display_name,
)
from pfc_shared.latex import write_table

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPOSITORY_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))

DIR_PARAMS = os.path.join(SCRIPT_DIR, "metrics", "parameters")
DIR_METRICS = os.path.join(SCRIPT_DIR, "metrics", "non_scalability")
HYPERPARAMETER_NAMES = {
    "variance_limit": "variance limit",
    "decay": "decay",
    "noise_distance": "noise distance",
}


def format_value(value) -> str:
    if pd.isna(value):
        return "-"
    if float(value).is_integer() or abs(value) >= 100:
        return f"{value:.0f}"
    return f"{value:.3g}"


def median_qcd(values: pd.Series) -> str:
    """Median and quartile coefficient of dispersion, $(Q_3 - Q_1)/(Q_3 + Q_1)$."""
    q1, median, q3 = values.quantile([0.25, 0.5, 0.75])
    qcd = (q3 - q1) / (q3 + q1) if q3 + q1 > 0 else np.nan
    dispersion = "-" if pd.isna(qcd) else f"{qcd:.2f}"
    return f"{format_value(median)} [{dispersion}]"


def hyperparameter_table(selected: pd.DataFrame, comparison):
    model = comparison.focus
    of_model = selected[selected["model"] == model].pivot_table(
        index=["dataset", "run"], columns="param_name", values="param_value"
    )
    names = [name for name in HYPERPARAMETER_NAMES if name in of_model.columns]
    datasets = [
        d for d in DATASET_ORDER if d in of_model.index.get_level_values("dataset")
    ]
    label = f"\\modelname{{{model_display_name(model)}}}"
    lines = [
        "Model & Hyperparameter & "
        + " & ".join(DATASET_SHORT_NAMES[d] for d in datasets)
        + r" \\",
        r"\midrule",
    ]
    for position, name in enumerate(names):
        parent = rf"\multirow{{{len(names)}}}{{*}}{{{label}}}" if position == 0 else ""
        cells = [median_qcd(of_model.loc[d, name]) for d in datasets]
        lines.append(" & ".join([parent, HYPERPARAMETER_NAMES[name], *cells]) + r" \\")
    title = f"Hyperparameters selected for {label}"
    out_dir = os.path.join(
        REPOSITORY_ROOT, "monografia", comparison.chapter, "artifacts", "parameters"
    )
    os.makedirs(out_dir, exist_ok=True)
    write_table(
        Path(out_dir) / f"{model}_parameters.tex",
        f"{title}: median [QCD] over the runs of the main experiment.",
        f"tab:params_{model.lower()}",
        "l l " + "c " * len(datasets),
        lines,
        short=title,
    )


def main():
    path_params = os.path.join(DIR_PARAMS, "best_params.parquet")
    path_metrics = os.path.join(DIR_METRICS, "run_level.parquet")

    if not os.path.exists(path_params) or not os.path.exists(path_metrics):
        print("Required input files not found.")
        return

    selected = pd.read_parquet(path_params)
    metrics = pd.read_parquet(path_metrics)
    selected = selected[selected["dataset"].isin(metrics["dataset"].unique())]

    for comparison in COMPARISONS:
        hyperparameter_table(selected, comparison)
        print(f"Generated the hyperparameter table of: {comparison.focus}")


if __name__ == "__main__":
    main()
