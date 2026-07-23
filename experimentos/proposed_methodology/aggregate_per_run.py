import glob
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

# Configuration
INPUT_DIR = "./results"
OUTPUT_DIR = "./aggregated_per_run"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Per-run point statistic (median) for these columns, PLUS an intra-run
# IQR-width (Q3 - Q1 across that run's own batches) restoring visibility
# into within-run batch-to-batch spread — reported downstream as its own
# separate metric/table rather than folded back into a mean+/-std-style
# pooled number (medians/quantiles have no "law of total variance"
# analogue to do that validly).
NAN_KEYS = [
    "ari_median_first_10",
    "ari_median_last_10",
    "ari_median_all",
    "ari_diff_max_last",
    "time_predict_ns_median",
    "time_predict_ns_iqr_width",
    "time_fit_ns_median",
    "time_fit_ns_iqr_width",
    "memory_bytes_median",
    "memory_bytes_iqr_width",
    "memory_bytes_max",
    "microcluster_count_median",
    "microcluster_count_iqr_width",
    "microcluster_count_max",
]


def iqr_width(series):
    return series.quantile(0.75) - series.quantile(0.25)


# Parse available files to build the processing map
all_files = glob.glob(os.path.join(INPUT_DIR, "*.csv"))
processing_map = {}

for filepath in all_files:
    filename = os.path.basename(filepath)
    # Filename structure: <dataset>_<model>_<run>.csv
    # e.g., chameleon_ds7_CluStream_0.csv
    name_without_ext = filename.replace(".csv", "")
    parts = name_without_ext.split("_")

    if len(parts) < 3:
        continue

    run_id = parts[-1]
    model_name = parts[-2]
    dataset_name = "_".join(parts[:-2])

    group_key = f"{dataset_name}_{model_name}"

    if group_key not in processing_map:
        processing_map[group_key] = []

    processing_map[group_key].append((run_id, filepath))

# Aggregation execution
for group_key, files in tqdm(
    processing_map.items(), desc="Aggregating models and datasets"
):
    group_results = []

    for run_id, filepath in files:
        try:
            df = pd.read_csv(filepath)

            # Check for evaluation errors
            if (df["status"] != "OK").any():
                group_results.append(
                    {
                        "run": int(run_id),
                        "run_status": "ERROR",
                        **{k: np.nan for k in NAN_KEYS},
                    }
                )
                continue

            # Compute ARI metrics
            ari_median_first_10 = df["ari"].head(10).median()
            ari_median_last_10 = df["ari"].tail(10).median()
            ari_median_all = df["ari"].median()
            ari_diff_max_last = df["ari"].max() - df["ari"].iloc[-1]

            # Compute Time metrics: per-run median (location) plus
            # intra-run IQR-width (within-run batch-to-batch spread)
            time_predict_median = df["time_predict_ns"].median()
            time_predict_iqr_width = iqr_width(df["time_predict_ns"])
            time_fit_median = df["time_fit_ns"].median()
            time_fit_iqr_width = iqr_width(df["time_fit_ns"])

            # Compute Memory metrics
            memory_median = df["memory_bytes"].median()
            memory_iqr_width = iqr_width(df["memory_bytes"])
            memory_max = df["memory_bytes"].max()

            # Compute Microcluster metrics
            mc_median = df["microcluster_count"].median()
            mc_iqr_width = iqr_width(df["microcluster_count"])
            mc_max = df["microcluster_count"].max()

            group_results.append(
                {
                    "run": int(run_id),
                    "run_status": "OK",
                    "ari_median_first_10": ari_median_first_10,
                    "ari_median_last_10": ari_median_last_10,
                    "ari_median_all": ari_median_all,
                    "ari_diff_max_last": ari_diff_max_last,
                    "time_predict_ns_median": time_predict_median,
                    "time_predict_ns_iqr_width": time_predict_iqr_width,
                    "time_fit_ns_median": time_fit_median,
                    "time_fit_ns_iqr_width": time_fit_iqr_width,
                    "memory_bytes_median": memory_median,
                    "memory_bytes_iqr_width": memory_iqr_width,
                    "memory_bytes_max": memory_max,
                    "microcluster_count_median": mc_median,
                    "microcluster_count_iqr_width": mc_iqr_width,
                    "microcluster_count_max": mc_max,
                }
            )

        except Exception as e:
            # Handle catastrophic I/O or parsing failure
            group_results.append(
                {
                    "run": int(run_id),
                    "run_status": f"I/O ERROR: {str(e)}",
                    **{k: np.nan for k in NAN_KEYS},
                }
            )

    # Sort sequentially by run index to enforce deterministic matrix structure
    df_out = pd.DataFrame(group_results).sort_values("run")

    # Serialize
    output_path = os.path.join(OUTPUT_DIR, f"{group_key}.csv")
    df_out.to_csv(output_path, index=False)
