#!/usr/bin/env python3
import os
import glob
import pandas as pd
import numpy as np

DIR_RESULTS = "./results"
DIR_PARAMS = "./results_params"
DIR_OUT = "./metrics"

DIR_OUT_NON_SCL = os.path.join(DIR_OUT, "non_scalability")
DIR_OUT_SCL = os.path.join(DIR_OUT, "scalability")
DIR_OUT_PARAMS = os.path.join(DIR_OUT, "parameters")

for d in [DIR_OUT_NON_SCL, DIR_OUT_SCL, DIR_OUT_PARAMS]:
    os.makedirs(d, exist_ok=True)


def aggregate_quantiles(
    df: pd.DataFrame, group_cols: list, val_cols: list
) -> pd.DataFrame:
    grouped = df.groupby(group_cols)[val_cols]
    aggregate_frames = []

    for column in val_cols:
        column_grouped = grouped[column]
        aggregate_frames.extend(
            [
                column_grouped.median().rename(f"{column}_median"),
                column_grouped.quantile(0.25).rename(f"{column}_q1"),
                column_grouped.quantile(0.75).rename(f"{column}_q3"),
            ]
        )

    return pd.concat(aggregate_frames, axis=1).reset_index()


def main():
    # 1. Parameter Aggregation
    param_files = glob.glob(os.path.join(DIR_PARAMS, "*.csv"))
    if param_files:
        df_params = pd.concat((pd.read_csv(f) for f in param_files), ignore_index=True)
        df_params.to_parquet(
            os.path.join(DIR_OUT_PARAMS, "best_params.parquet"), index=False
        )

    # 2. Result Parsing and Partitioning
    result_files = glob.glob(os.path.join(DIR_RESULTS, "*.csv"))

    non_scl_records = []
    scl_records = []

    for file in result_files:
        basename = os.path.basename(file).replace(".csv", "")
        # Format: {dataset}_{model}_{run}.csv
        parts = basename.split("_")
        run = int(parts[-1])
        model = parts[-2]
        dataset = "_".join(parts[:-2])

        df = pd.read_csv(file)

        # Filter out trials that failed during execution
        if not df[df["status"].str.startswith("ERROR", na=False)].empty:
            continue

        if len(df) < 5:
            continue  # Insufficient batches for 5th/penultimate metric extraction

        is_scl = dataset.startswith("SCL-")

        if is_scl:
            df["dataset"] = dataset
            df["model"] = model
            df["run"] = run
            df["dim"] = int(dataset.split("-")[1])
            scl_records.append(df)
        else:
            ari_array = df["ari"].to_numpy()
            mem_array = df["memory_bytes"].to_numpy()
            mc_array = df["microcluster_count"].to_numpy()

            non_scl_records.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "run": run,
                    "ari_penultimate": ari_array[-2],
                    "ari_fifth": ari_array[4],
                    "ari_diff": np.max(ari_array) - ari_array[-1],
                    "time_predict_cum_ms": df["time_predict_wall_ns"].sum() / 1e6,
                    "time_fit_cum_ms": df["time_fit_wall_ns"].sum() / 1e6,
                    "memory_max_bytes": np.max(mem_array),
                    "memory_penultimate_bytes": mem_array[-2],
                    "mc_max": np.max(mc_array),
                    "mc_penultimate": mc_array[-2],
                }
            )

    # 3. Non-Scalability Processing
    if non_scl_records:
        df_non_scl = pd.DataFrame(non_scl_records)
        df_non_scl.to_parquet(
            os.path.join(DIR_OUT_NON_SCL, "run_level.parquet"), index=False
        )

        val_columns = [
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
        df_non_scl_agg = aggregate_quantiles(
            df_non_scl, ["dataset", "model"], val_columns
        )
        df_non_scl_agg.to_parquet(
            os.path.join(DIR_OUT_NON_SCL, "aggregated.parquet"), index=False
        )

    # 4. Scalability Processing
    if scl_records:
        df_scl = pd.concat(scl_records, ignore_index=True)
        df_scl["time_predict_ms"] = df_scl["time_predict_wall_ns"] / 1e6
        df_scl["time_fit_ms"] = df_scl["time_fit_wall_ns"] / 1e6

        # Drop raw ns columns to save space
        df_scl.drop(
            columns=[
                "time_predict_wall_ns",
                "time_fit_wall_ns",
                "time_predict_cpu_ns",
                "time_fit_cpu_ns",
                "status",
            ],
            inplace=True,
        )

        df_scl.to_parquet(os.path.join(DIR_OUT_SCL, "run_level.parquet"), index=False)

        val_columns = [
            "ari",
            "time_predict_ms",
            "time_fit_ms",
            "memory_bytes",
            "microcluster_count",
        ]
        df_scl_agg = aggregate_quantiles(
            df_scl, ["dataset", "dim", "model", "batch"], val_columns
        )
        df_scl_agg.to_parquet(
            os.path.join(DIR_OUT_SCL, "aggregated_by_batch.parquet"), index=False
        )


if __name__ == "__main__":
    main()
