#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np

from editorial_config import DATASET_ORDER, MODEL_ORDER, dataset_display_name

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPOSITORY_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))

DIR_PARAMS = os.path.join(SCRIPT_DIR, "metrics", "parameters")
DIR_METRICS = os.path.join(SCRIPT_DIR, "metrics", "non_scalability")
DIR_OUT = os.path.join(
    REPOSITORY_ROOT,
    "monografia",
    "MethodologyReassessment",
    "artifacts",
    "parameters",
)
os.makedirs(DIR_OUT, exist_ok=True)


def format_model_name(model: str) -> str:
    return f"\\modelname{{{model}}}"


def format_dataset_name(dataset: str) -> str:
    escaped_dataset = dataset_display_name(dataset).replace("_", "\\_")
    return f"\\datasetname{{{escaped_dataset}}}"


def format_val(val) -> str:
    if pd.isna(val):
        return "-"
    if isinstance(val, (float, np.floating)):
        if val.is_integer():
            return f"{int(val)}"
        return f"{val:.4g}"
    return str(val)


def process_model_parameters(df_params: pd.DataFrame, model: str) -> dict:
    df_m_params_long = df_params[df_params["model"] == model]
    if df_m_params_long.empty:
        return {}

    try:
        df_m_params = df_m_params_long.pivot_table(
            index=["dataset", "model", "run"],
            columns="param_name",
            values="param_value",
        ).reset_index()
    except KeyError:
        df_m_params = df_m_params_long

    meta_cols = {"dataset", "model", "run"}
    param_cols = [c for c in df_m_params.columns if c not in meta_cols]

    if not param_cols:
        return {}

    for p in param_cols:
        df_m_params[p] = pd.to_numeric(df_m_params[p], errors="coerce")

    latex_lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{Hyperparameter dispersion for {model}: median $(Q_1, Q_3)$ [QCD].}}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{l " + "c " * len(param_cols) + "}",
        "\\toprule",
        "Dataset & "
        + " & ".join([p.replace("_", "\\_") for p in param_cols])
        + " \\\\",
        "\\midrule",
    ]

    dataset_order = {dataset: index for index, dataset in enumerate(DATASET_ORDER)}
    datasets = sorted(
        df_m_params["dataset"].unique(),
        key=lambda dataset: dataset_order.get(dataset, len(DATASET_ORDER)),
    )
    qcd_sums = {}

    for dataset in datasets:
        df_dp = df_m_params[df_m_params["dataset"] == dataset]
        row_str = format_dataset_name(dataset)
        sum_qcd = 0.0

        for p in param_cols:
            med = df_dp[p].median()
            q1 = df_dp[p].quantile(0.25)
            q3 = df_dp[p].quantile(0.75)

            denom = q3 + q1
            if pd.notna(denom) and denom > 0:
                qcd = (q3 - q1) / denom
                sum_qcd += qcd
                qcd_str = f" [{qcd:.3f}]"
            else:
                qcd_str = " [-]"

            row_str += (
                f" & {format_val(med)} ({format_val(q1)}, {format_val(q3)}){qcd_str}"
            )

        latex_lines.append(row_str + " \\\\")
        qcd_sums[dataset] = sum_qcd

    latex_lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "}",
            f"\\label{{tab:params_{model.lower()}}}",
            "\\end{table}",
        ]
    )

    with open(os.path.join(DIR_OUT, f"{model}_parameters.tex"), "w") as f:
        f.write("\n".join(latex_lines))
        f.write("\n")

    return qcd_sums


def generate_qcd_sum_table(all_qcd_sums: dict, models: list):
    df_qcd = pd.DataFrame(all_qcd_sums)

    latex_lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Unified sum of Quartile Coefficients of Dispersion per model. Bold indicates the absolute deterministic minimum, and red indicates the absolute deterministic maximum.}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{l " + "c " * len(models) + "}",
        "\\toprule",
        "Dataset & "
        + " & ".join(format_model_name(model) for model in models)
        + " \\\\",
        "\\midrule",
    ]

    for dataset in df_qcd.index:
        row = df_qcd.loc[dataset]
        min_val = row.min()
        max_val = row.max()

        row_str = format_dataset_name(dataset)
        for m in models:
            val = row.get(m, np.nan)
            if pd.isna(val):
                row_str += " & -"
            else:
                val_str = f"{val:.3f}"
                if val == min_val and val == max_val:
                    row_str += f" & {val_str}"
                elif val == min_val:
                    row_str += f" & \\textbf{{{val_str}}}"
                elif val == max_val:
                    row_str += f" & \\textcolor{{red}}{{{val_str}}}"
                else:
                    row_str += f" & {val_str}"
        latex_lines.append(row_str + " \\\\")

    latex_lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "}",
            "\\label{tab:qcd_sum}",
            "\\end{table}",
        ]
    )

    with open(os.path.join(DIR_OUT, "qcd_sum.tex"), "w") as f:
        f.write("\n".join(latex_lines))
        f.write("\n")


def main():
    path_params = os.path.join(DIR_PARAMS, "best_params.parquet")
    path_metrics = os.path.join(DIR_METRICS, "run_level.parquet")

    if not os.path.exists(path_params) or not os.path.exists(path_metrics):
        print("Required input files not found.")
        return

    df_params = pd.read_parquet(path_params)
    df_metrics = pd.read_parquet(path_metrics)

    valid_datasets = df_metrics["dataset"].unique()
    df_params = df_params[df_params["dataset"].isin(valid_datasets)]

    models = [model for model in MODEL_ORDER if model in df_params["model"].unique()]
    all_qcd_sums = {}

    for model in models:
        sums = process_model_parameters(df_params, model)
        if sums:
            all_qcd_sums[model] = sums
            print(f"Generated parameter artifact for: {model}")

    generate_qcd_sum_table(all_qcd_sums, models)
    print("Generated unified QCD sum table.")


if __name__ == "__main__":
    main()
