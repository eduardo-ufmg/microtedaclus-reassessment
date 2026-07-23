import glob
import os
from collections import Counter

import pandas as pd

# Aggregates the per-run hyperparameter files written by experiment.py
# (./results_params/{dataset}_{model}_{run}.csv, long format:
# dataset,model,run,param_name,param_value) into one summary table per
# model: rows = dataset, columns = parameter name.
#
# Different models have different search spaces, so — unlike the
# performance metrics in aggregate_runs.py — a single dataset x model
# pivot per parameter doesn't apply; output is one file per model instead.

INPUT_DIR = "./results_params"
OUTPUT_DIR = "./metrics_summary/params"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def format_cell(raw_values):
    """raw_values: list of param_value strings (as read back from CSV) for
    one (dataset, model, param_name) group across runs. Numeric params are
    summarized as median [Q1, Q3]; non-numeric (categorical) params by
    their mode."""
    numeric = pd.to_numeric(pd.Series(raw_values), errors="coerce")

    if numeric.notna().all():
        vals = numeric.tolist()
        med = numeric.median()
        q1 = numeric.quantile(0.25)
        q3 = numeric.quantile(0.75)
        if all(float(v).is_integer() for v in vals):
            return f"{int(med)} [{int(q1)}, {int(q3)}]"
        return f"{med:.4g} [{q1:.4g}, {q3:.4g}]"

    top, cnt = Counter(str(v) for v in raw_values).most_common(1)[0]
    return f"{top} ({cnt}/{len(raw_values)})"


def main():
    all_files = glob.glob(os.path.join(INPUT_DIR, "*.csv"))
    if not all_files:
        print("No parameter files found — nothing to aggregate.")
        return

    df_list = []
    for filepath in all_files:
        try:
            df = pd.read_csv(filepath)
            df_list.append(df)
        except Exception as e:
            print(f"Skipping {filepath} due to parsing error: {e}")

    if not df_list:
        print("No valid parameter data found.")
        return

    master_df = pd.concat(df_list, ignore_index=True)

    for model in sorted(master_df["model"].unique()):
        model_df = master_df[master_df["model"] == model]

        records = []
        for (dataset, param_name), group in model_df.groupby(["dataset", "param_name"]):
            records.append(
                {
                    "dataset": dataset,
                    "param_name": param_name,
                    "value": format_cell(group["param_value"].tolist()),
                }
            )

        table = pd.DataFrame(records).pivot(
            index="dataset", columns="param_name", values="value"
        )
        table.sort_index(inplace=True)

        output_path = os.path.join(OUTPUT_DIR, f"{model}.csv")
        table.to_csv(output_path)
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
