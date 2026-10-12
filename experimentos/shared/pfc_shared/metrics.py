"""Run-level metrics of one result file of the evaluation protocol."""

import numpy as np
import pandas as pd


def is_valid_result(df: pd.DataFrame) -> bool:
    """A result file enters the analysis if no batch failed and it has at least five batches."""
    return df[df["status"].str.startswith("ERROR", na=False)].empty and len(df) >= 5


def add_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Per-batch wall-clock times in milliseconds."""
    df["time_predict_ms"] = df["time_predict_wall_ns"] / 1e6
    df["time_fit_ms"] = df["time_fit_wall_ns"] / 1e6
    return df


def scalar_metrics(df: pd.DataFrame) -> dict:
    """Run-level metrics of one result file (after add_time_columns)."""
    ari_array = df["ari"].to_numpy()
    mem_array = df["memory_bytes"].to_numpy()
    mc_array = df["microcluster_count"].to_numpy()

    return {
        "ari_overall": np.median(ari_array),
        "ari_steadystate": ari_array[-2],
        "ari_fifth": ari_array[4],
        "ari_diff": np.max(ari_array) - ari_array[-2],
        "time_predict_cum_ms": df["time_predict_ms"].sum(),
        "time_fit_cum_ms": df["time_fit_ms"].sum(),
        "memory_max_bytes": np.max(mem_array),
        "memory_steadystate_bytes": mem_array[-2],
        "mc_max": np.max(mc_array),
        "mc_steadystate": mc_array[-2],
    }
