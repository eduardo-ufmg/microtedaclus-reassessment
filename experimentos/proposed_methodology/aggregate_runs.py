import glob
import os

import numpy as np
import pandas as pd

# Configuration
INPUT_DIR = "./aggregated_per_run"
OUTPUT_DIR = "./metrics_summary"
os.makedirs(OUTPUT_DIR, exist_ok=True)

KNOWN_MODELS = [
    "CluStream",
    "DenStream",
    "MicroTEDAClus",
    "CEDAS",
    "TEDACloud",
    "AutoCloud",
    "StreamKM",
]


def parse_filename(filename):
    """Extracts dataset and model name from <dataset>_<model>.csv."""
    name_without_ext = filename.replace(".csv", "")
    for model in KNOWN_MODELS:
        if name_without_ext.endswith(f"_{model}"):
            dataset = name_without_ext[: -(len(model) + 1)]
            return dataset, model
    raise ValueError(f"Unrecognized model in filename: {filename}")


def format_cell(median_val, q1_val, q3_val):
    """Formats the median/IQR triple into a single cell string."""
    if pd.isna(median_val) or pd.isna(q1_val) or pd.isna(q3_val):
        return "NaN"
    return f"{median_val:.4g} [{q1_val:.4g}, {q3_val:.4g}]"


# 1. Ingest Data
all_files = glob.glob(os.path.join(INPUT_DIR, "*.csv"))
raw_data: list[pd.DataFrame] = []

for filepath in all_files:
    filename = os.path.basename(filepath)
    try:
        dataset, model = parse_filename(filename)
    except ValueError:
        continue

    df = pd.read_csv(filepath)
    df_valid = df[df["run_status"] == "OK"].copy()

    if df_valid.empty:
        continue

    df_valid["dataset"] = dataset
    df_valid["model"] = model
    raw_data.append(df_valid)

if not raw_data:
    print("No valid runs found for aggregation.")
    exit(0)

master_df = pd.concat(raw_data, ignore_index=True)
datasets = sorted(master_df["dataset"].unique())
models = sorted(master_df["model"].unique())

# 2. Identify Metric Columns
# Base columns exclude structural identifiers. Each column in
# aggregated_per_run/*.csv is already a single per-run point statistic
# (e.g. a per-run median), so no paired/unpaired split is needed here:
# medians and quantiles have no "law of total variance" analogue to pool
# a mean+std pair, so every metric is treated uniformly as an inter-run
# distribution of per-run values.
all_cols = set(master_df.columns) - {"run", "run_status", "dataset", "model"}

# 3. Statistical Aggregation (inter-run median / Q1 / Q3)
aggregated_results = {metric: [] for metric in all_cols}

grouped = master_df.groupby(["dataset", "model"])

for (dataset, model), group in grouped:
    for metric in all_cols:
        vals = group[metric]
        inter_median = vals.median()
        inter_q1 = vals.quantile(0.25)
        inter_q3 = vals.quantile(0.75)

        aggregated_results[metric].append(
            {
                "dataset": dataset,
                "model": model,
                "value": format_cell(inter_median, inter_q1, inter_q3),
            }
        )

# 4. Matrix Serialization
print("Serializing aggregated metric matrices...")
for metric, records in aggregated_results.items():
    if not records:
        continue

    df_metric = pd.DataFrame(records)

    # Pivot: Rows = dataset, Columns = model
    pivot_df = df_metric.pivot(index="dataset", columns="model", values="value")

    # Ensure all models are present as columns even if they failed completely
    for m in KNOWN_MODELS:
        if m not in pivot_df.columns:
            pivot_df[m] = "NaN"

    # Sort index and enforce column order
    pivot_df.sort_index(inplace=True)
    pivot_df = pivot_df[KNOWN_MODELS]

    # Save
    output_path = os.path.join(OUTPUT_DIR, f"{metric}.csv")
    pivot_df.to_csv(output_path)

print(f"Aggregation complete. Output stored in {OUTPUT_DIR}/")
