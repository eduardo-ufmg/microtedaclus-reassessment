import os
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm

# Configuration
INPUT_DIR = "./results"
OUTPUT_DIR = "./aggregated_per_run"
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
                        "ari_mean_first_10": np.nan,
                        "ari_mean_last_10": np.nan,
                        "ari_mean_all": np.nan,
                        "ari_diff_max_last": np.nan,
                        "time_predict_ns_mean": np.nan,
                        "time_predict_ns_std": np.nan,
                        "time_fit_ns_mean": np.nan,
                        "time_fit_ns_std": np.nan,
                        "memory_bytes_mean": np.nan,
                        "memory_bytes_std": np.nan,
                        "memory_bytes_max": np.nan,
                        "microcluster_count_mean": np.nan,
                        "microcluster_count_std": np.nan,
                        "microcluster_count_max": np.nan,
                    }
                )
                continue

            # Compute ARI metrics
            ari_mean_first_10 = df["ari"].head(10).mean()
            ari_mean_last_10 = df["ari"].tail(10).mean()
            ari_mean_all = df["ari"].mean()
            ari_diff_max_last = df["ari"].max() - df["ari"].iloc[-1]

            # Compute Time metrics
            time_predict_mean = df["time_predict_ns"].mean()
            time_predict_std = df["time_predict_ns"].std()
            time_fit_mean = df["time_fit_ns"].mean()
            time_fit_std = df["time_fit_ns"].std()

            # Compute Memory metrics
            memory_mean = df["memory_bytes"].mean()
            memory_std = df["memory_bytes"].std()
            memory_max = df["memory_bytes"].max()

            # Compute Microcluster metrics
            mc_mean = df["microcluster_count"].mean()
            mc_std = df["microcluster_count"].std()
            mc_max = df["microcluster_count"].max()

            group_results.append(
                {
                    "run": int(run_id),
                    "run_status": "OK",
                    "ari_mean_first_10": ari_mean_first_10,
                    "ari_mean_last_10": ari_mean_last_10,
                    "ari_mean_all": ari_mean_all,
                    "ari_diff_max_last": ari_diff_max_last,
                    "time_predict_ns_mean": time_predict_mean,
                    "time_predict_ns_std": time_predict_std,
                    "time_fit_ns_mean": time_fit_mean,
                    "time_fit_ns_std": time_fit_std,
                    "memory_bytes_mean": memory_mean,
                    "memory_bytes_std": memory_std,
                    "memory_bytes_max": memory_max,
                    "microcluster_count_mean": mc_mean,
                    "microcluster_count_std": mc_std,
                    "microcluster_count_max": mc_max,
                }
            )

        except Exception as e:
            # Handle catastrophic I/O or parsing failure
            group_results.append(
                {
                    "run": int(run_id),
                    "run_status": f"I/O ERROR: {str(e)}",
                    **{
                        k: np.nan
                        for k in [
                            "ari_mean_first_10",
                            "ari_mean_last_10",
                            "ari_mean_all",
                            "ari_diff_max_last",
                            "time_predict_ns_mean",
                            "time_predict_ns_std",
                            "time_fit_ns_mean",
                            "time_fit_ns_std",
                            "memory_bytes_mean",
                            "memory_bytes_std",
                            "memory_bytes_max",
                            "microcluster_count_mean",
                            "microcluster_count_std",
                            "microcluster_count_max",
                        ]
                    },
                }
            )

    # Sort sequentially by run index to enforce deterministic matrix structure
    df_out = pd.DataFrame(group_results).sort_values("run")

    # Serialize
    output_path = os.path.join(OUTPUT_DIR, f"{group_key}.csv")
    df_out.to_csv(output_path, index=False)
