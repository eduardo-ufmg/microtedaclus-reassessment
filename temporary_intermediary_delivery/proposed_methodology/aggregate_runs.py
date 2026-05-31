import os
import glob
import pandas as pd
import numpy as np

# Configuration
INPUT_DIR = "./aggregated_per_run"
OUTPUT_DIR = "./metrics_summary"
os.makedirs(OUTPUT_DIR, exist_ok=True)

KNOWN_MODELS = ["CluStream", "DenStream", "MicroTEDAClus"]


def parse_filename(filename):
    """Extracts dataset and model name from <dataset>_<model>.csv."""
    name_without_ext = filename.replace(".csv", "")
    for model in KNOWN_MODELS:
        if name_without_ext.endswith(f"_{model}"):
            dataset = name_without_ext[: -(len(model) + 1)]
            return dataset, model
    raise ValueError(f"Unrecognized model in filename: {filename}")


def format_cell(mean_val, std_val):
    """Formats the statistical pair into a single cell string."""
    if pd.isna(mean_val) or pd.isna(std_val):
        return "NaN"
    return f"{mean_val:.4g} \u00b1 {std_val:.4g}"


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

# 2. Identify Metric Types
# Base columns exclude structural identifiers
all_cols = set(master_df.columns) - {"run", "run_status", "dataset", "model"}
std_cols = {col for col in all_cols if col.endswith("_std")}
mean_cols = {col for col in all_cols if col.endswith("_mean")}

# Paired metrics (require Law of Total Variance) e.g., 'time_fit_ns'
paired_metrics = {
    col.replace("_mean", "")
    for col in mean_cols
    if f"{col.replace('_mean', '')}_std" in std_cols
}

# Unpaired metrics (require standard Inter-Run stats) e.g., 'ari_mean_all', 'memory_bytes_max'
unpaired_metrics = (
    all_cols
    - {f"{m}_mean" for m in paired_metrics}
    - {f"{m}_std" for m in paired_metrics}
)

# 3. Statistical Aggregation
aggregated_results = {
    metric: [] for metric in list(paired_metrics) + list(unpaired_metrics)
}

grouped = master_df.groupby(["dataset", "model"])

for (dataset, model), group in grouped:
    # Process Paired Metrics (Law of Total Variance)
    for metric in paired_metrics:
        run_means = group[f"{metric}_mean"]
        run_stds = group[f"{metric}_std"]

        # Pooled Mean: Mean of means
        pooled_mean = run_means.mean()

        # Pooled Variance: Mean of variances + Variance of means
        mean_of_vars = (run_stds**2).mean()
        var_of_means = run_means.var(ddof=1) if len(run_means) > 1 else 0.0

        pooled_std = np.sqrt(mean_of_vars + var_of_means)

        aggregated_results[metric].append(
            {
                "dataset": dataset,
                "model": model,
                "value": format_cell(pooled_mean, pooled_std),
            }
        )

    # Process Unpaired Metrics (Standard Inter-Run Stats)
    for metric in unpaired_metrics:
        vals = group[metric]
        inter_mean = vals.mean()
        inter_std = vals.std(ddof=1) if len(vals) > 1 else 0.0

        aggregated_results[metric].append(
            {
                "dataset": dataset,
                "model": model,
                "value": format_cell(inter_mean, inter_std),
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
