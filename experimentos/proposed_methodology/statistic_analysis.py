#!/usr/bin/env python3
import os
import pandas as pd

from editorial_config import SUBSETS, effective_models
from pfc_shared.statistics import METRICS, process_metric

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DIR_METRICS_NON_SCL = os.path.join(SCRIPT_DIR, "metrics", "non_scalability")
DIR_OUT = os.path.join(SCRIPT_DIR, "statistical_analysis")


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

            models = effective_models(subset_name, metric)
            if len(models) < 2:
                continue

            df_table = process_metric(df_runs, metric, config, models)
            if df_table.empty:
                continue

            out_path = os.path.join(
                subset_out_dir, f"{metric}_statistical_analysis.csv"
            )
            df_table.to_csv(out_path)
        print(f"Generated statistical relevance tables for subset: {subset_name}")


if __name__ == "__main__":
    main()
