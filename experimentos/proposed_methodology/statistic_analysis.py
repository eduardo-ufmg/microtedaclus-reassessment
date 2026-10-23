#!/usr/bin/env python3
import os
import pandas as pd

from editorial_config import COMPARISONS, effective_models
from pfc_shared.statistics import METRICS, compare_with_focus

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DIR_METRICS_NON_SCL = os.path.join(SCRIPT_DIR, "metrics", "non_scalability")
DIR_OUT = os.path.join(SCRIPT_DIR, "statistical_analysis")


def main():
    path_in = os.path.join(DIR_METRICS_NON_SCL, "run_level.parquet")
    if not os.path.exists(path_in):
        print(f"File not found: {path_in}")
        return

    df_runs = pd.read_parquet(path_in)

    for comparison in COMPARISONS:
        out_dir = os.path.join(DIR_OUT, comparison.name)
        os.makedirs(out_dir, exist_ok=True)

        for metric in METRICS:
            if metric not in df_runs.columns:
                continue

            focus, *others = effective_models(comparison, metric)
            table = compare_with_focus(df_runs, metric, focus, others)
            if table.empty:
                continue

            table.to_csv(os.path.join(out_dir, f"{metric}.csv"), index=False)
        print(f"Compared the models of: {comparison.name}")


if __name__ == "__main__":
    main()
