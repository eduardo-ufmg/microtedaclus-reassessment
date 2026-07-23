import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Configuration
INPUT_DIR = "./results"
OUTPUT_DIR = "./plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define scaling factors, axis scales, and logical bounds
# You may adjust the 'y_scale' ('linear' or 'log') as needed after visual inspection.
METRIC_CONFIG = {
    "ari": {
        "source_col": "ari",
        "scale": 1.0,
        "y_scale": "linear",
        "min": -1.0,
        "max": 1.0,
        "ylabel": "Adjusted Rand Index",
    },
    "time_predict_ms": {
        "source_col": "time_predict_ns",
        "scale": 1e-6,
        "y_scale": "log",
        "min": 0.0,
        "max": np.inf,
        "ylabel": "Prediction Time (ms)",
    },
    "time_fit_ms": {
        "source_col": "time_fit_ns",
        "scale": 1e-6,
        "y_scale": "log",
        "min": 0.0,
        "max": np.inf,
        "ylabel": "Fit Time (ms)",
    },
    "memory_mb": {
        "source_col": "memory_bytes",
        "scale": 1 / (1024**2),
        "y_scale": "linear",
        "min": 0.0,
        "max": np.inf,
        "ylabel": "Memory Footprint (MB)",
    },
    "microcluster_count": {
        "source_col": "microcluster_count",
        "scale": 1.0,
        "y_scale": "linear",
        "min": 0.0,
        "max": np.inf,
        "ylabel": "Microcluster Count",
    },
}


def main():
    print("1. Ingesting raw evaluation telemetry...")
    all_files = glob.glob(os.path.join(INPUT_DIR, "*.csv"))
    df_list: list[pd.DataFrame] = []

    for filepath in all_files:
        filename = os.path.basename(filepath)
        name_without_ext = filename.replace(".csv", "")
        name_without_parentheses = name_without_ext.replace("(", "").replace(")", "")
        parts = name_without_parentheses.split("_")

        if len(parts) < 3:
            continue

        run_id = parts[-1]
        model_name = parts[-2]
        dataset_name = "_".join(parts[:-2])

        try:
            df = pd.read_csv(filepath)
            # Filter explicitly for successful batches
            df = df[df["status"] == "OK"].copy()

            if df.empty:
                continue

            df["dataset"] = dataset_name
            df["model"] = model_name
            df["run"] = int(run_id)
            df_list.append(df)
        except Exception as e:
            print(f"Skipping {filepath} due to parsing error: {e}")

    if not df_list:
        print("No valid data found to plot. Exiting.")
        exit(0)

    master_df = pd.concat(df_list, ignore_index=True)

    # Apply scalers
    for target_metric, config in METRIC_CONFIG.items():
        master_df[target_metric] = master_df[config["source_col"]] * config["scale"]

    # Extract global entities
    datasets = master_df["dataset"].unique()
    models = sorted(master_df["model"].unique())

    # Consistent color mapping for models across all plots
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, len(models)))
    color_map = {model: color for model, color in zip(models, colors)}

    print("2. Computing aggregations and generating plots...")
    # Aggregation: Group by dataset, model, and batch index
    agg_cols = list(METRIC_CONFIG.keys())
    grouped = (
        master_df.groupby(["dataset", "model", "batch"])[agg_cols]
        .agg(
            [
                "median",
                lambda s: s.quantile(0.25),
                lambda s: s.quantile(0.75),
            ]
        )
        .reset_index()
    )
    # Name the quantile lambdas explicitly (pandas otherwise assigns
    # "<lambda_0>"/"<lambda_1>", which is not stable/readable downstream).
    grouped.columns = pd.MultiIndex.from_tuples(
        [
            (
                (col, {"<lambda_0>": "q1", "<lambda_1>": "q3"}.get(stat, stat))
                if stat
                else (col, stat)
            )
            for col, stat in grouped.columns
        ]
    )

    for dataset in datasets:
        ds_data = grouped[grouped["dataset"] == dataset]

        for metric, config in METRIC_CONFIG.items():
            fig, ax = plt.subplots(figsize=(10, 6))

            for model in models:
                model_data = ds_data[ds_data["model"] == model].sort_values("batch")
                if model_data.empty:
                    continue

                batches = model_data["batch"].values
                medians = model_data[(metric, "median")].values
                lower_bound = model_data[(metric, "q1")].values
                upper_bound = model_data[(metric, "q3")].values
                # Unlike a mean+/-std band, Q1/Q3 need no NaN patching for
                # single-surviving-run batches (quantile of one value is
                # that value) and are already within [min, max] by
                # construction, so no clipping is needed either.

                # Log scale safety: matplotlib fill_between fails if lower_bound <= 0 on a log axis
                logical_min = config["min"]
                if config["y_scale"] == "log" and logical_min <= 0:
                    logical_min = np.min(medians)
                lower_bound = np.clip(lower_bound, logical_min, config["max"])

                ax.plot(
                    batches, medians, label=model, color=color_map[model], linewidth=2.0
                )
                ax.fill_between(
                    batches, lower_bound, upper_bound, color=color_map[model], alpha=0.2
                )

            ax.set_title(f"{dataset} - {config['ylabel']}")
            ax.set_xlabel("Batch Index")
            ax.set_ylabel(config["ylabel"])
            ax.set_yscale(config["y_scale"])
            ax.legend(title="Models", loc="best")
            ax.grid(True, which="both", linestyle="--", alpha=0.5)

            plt.tight_layout()
            plot_path = os.path.join(OUTPUT_DIR, f"{dataset}_{metric}.png")
            plt.savefig(plot_path, dpi=300)
            plt.close(fig)

    print(f"Processing complete. Plots serialized to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
