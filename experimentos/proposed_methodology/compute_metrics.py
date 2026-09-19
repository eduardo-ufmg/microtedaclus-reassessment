#!/usr/bin/env python3
import os
import glob
import pandas as pd
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DIR_RESULTS = os.path.join(SCRIPT_DIR, "results")
DIR_PARAMS = os.path.join(SCRIPT_DIR, "results_params")
DIR_OUT_NON_SCL = os.path.join(SCRIPT_DIR, "metrics", "non_scalability")
DIR_OUT_SCL = os.path.join(SCRIPT_DIR, "metrics", "scalability")
DIR_OUT_PARAMS = os.path.join(SCRIPT_DIR, "metrics", "parameters")

for d in [DIR_OUT_NON_SCL, DIR_OUT_SCL, DIR_OUT_PARAMS]:
    os.makedirs(d, exist_ok=True)


def aggregate_quantiles(
    df: pd.DataFrame, group_cols: list, val_cols: list
) -> pd.DataFrame:
    grouped = df.groupby(group_cols)[val_cols]
    median_df = grouped.median().rename(columns=lambda column: f"{column}_median")
    q1_df = grouped.quantile(0.25).rename(columns=lambda column: f"{column}_q1")
    q3_df = grouped.quantile(0.75).rename(columns=lambda column: f"{column}_q3")
    agg_df = pd.concat([median_df, q1_df, q3_df], axis=1)
    return agg_df.reset_index()


def main():
    param_files = glob.glob(os.path.join(DIR_PARAMS, "*.csv"))
    if param_files:
        df_params = pd.concat((pd.read_csv(f) for f in param_files), ignore_index=True)
        df_params.to_parquet(
            os.path.join(DIR_OUT_PARAMS, "best_params.parquet"), index=False
        )

    result_files = glob.glob(os.path.join(DIR_RESULTS, "*.csv"))

    non_scl_scalar_records = []
    non_scl_batch_records = []
    scl_records = []

    for file in result_files:
        basename = os.path.basename(file).replace(".csv", "")
        parts = basename.split("_")
        run = int(parts[-1])
        model = parts[-2]
        dataset = "_".join(parts[:-2])

        df = pd.read_csv(file)
        if not df[df["status"].str.startswith("ERROR", na=False)].empty or len(df) < 5:
            continue

        df["dataset"] = dataset
        df["model"] = model
        df["run"] = run
        df["time_predict_ms"] = df["time_predict_wall_ns"] / 1e6
        df["time_fit_ms"] = df["time_fit_wall_ns"] / 1e6

        if dataset.startswith("SCL-"):
            df["dim"] = int(dataset.split("-")[1])
            scl_records.append(
                df[
                    [
                        "dataset",
                        "dim",
                        "model",
                        "run",
                        "batch",
                        "ari",
                        "time_predict_ms",
                        "time_fit_ms",
                        "memory_bytes",
                        "microcluster_count",
                    ]
                ]
            )
        else:
            non_scl_batch_records.append(
                df[
                    [
                        "dataset",
                        "model",
                        "run",
                        "batch",
                        "ari",
                        "time_predict_ms",
                        "time_fit_ms",
                        "memory_bytes",
                        "microcluster_count",
                    ]
                ]
            )

            ari_array = df["ari"].to_numpy()
            mem_array = df["memory_bytes"].to_numpy()
            mc_array = df["microcluster_count"].to_numpy()

            non_scl_scalar_records.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "run": run,
                    "ari_overall": np.median(ari_array),
                    "ari_penultimate": ari_array[-2],
                    "ari_fifth": ari_array[4],
                    "ari_diff": np.max(ari_array) - ari_array[-1],
                    "time_predict_cum_ms": df["time_predict_ms"].sum(),
                    "time_fit_cum_ms": df["time_fit_ms"].sum(),
                    "memory_max_bytes": np.max(mem_array),
                    "memory_penultimate_bytes": mem_array[-2],
                    "mc_max": np.max(mc_array),
                    "mc_penultimate": mc_array[-2],
                }
            )

    val_columns = [
        "ari",
        "time_predict_ms",
        "time_fit_ms",
        "memory_bytes",
        "microcluster_count",
    ]

    if non_scl_scalar_records:
        # Scalar export
        df_non_scl = pd.DataFrame(non_scl_scalar_records)
        df_non_scl.to_parquet(
            os.path.join(DIR_OUT_NON_SCL, "run_level.parquet"), index=False
        )
        scalar_cols = [
            "ari_overall",
            "ari_penultimate",
            "ari_fifth",
            "ari_diff",
            "time_predict_cum_ms",
            "time_fit_cum_ms",
            "memory_max_bytes",
            "memory_penultimate_bytes",
            "mc_max",
            "mc_penultimate",
        ]
        aggregate_quantiles(df_non_scl, ["dataset", "model"], scalar_cols).to_parquet(
            os.path.join(DIR_OUT_NON_SCL, "aggregated.parquet"), index=False
        )

        # Batch-level export for internal plots
        df_non_scl_batch = pd.concat(non_scl_batch_records, ignore_index=True)
        aggregate_quantiles(
            df_non_scl_batch, ["dataset", "model", "batch"], val_columns
        ).to_parquet(
            os.path.join(DIR_OUT_NON_SCL, "aggregated_by_batch.parquet"), index=False
        )

    if scl_records:
        df_scl = pd.concat(scl_records, ignore_index=True)
        df_scl.to_parquet(os.path.join(DIR_OUT_SCL, "run_level.parquet"), index=False)
        aggregate_quantiles(
            df_scl, ["dataset", "dim", "model", "batch"], val_columns
        ).to_parquet(
            os.path.join(DIR_OUT_SCL, "aggregated_by_batch.parquet"), index=False
        )


if __name__ == "__main__":
    main()
