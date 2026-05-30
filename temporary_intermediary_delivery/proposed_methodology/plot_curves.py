import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

print("1. Ingesting raw evaluation telemetry...")
all_files = glob.glob(os.path.join(INPUT_DIR, "*.csv"))
df_list = []

for filepath in all_files:
    filename = os.path.basename(filepath)
    name_without_ext = filename.replace(".csv", "")
    parts = name_without_ext.split("_")

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
colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
color_map = {model: color for model, color in zip(models, colors)}

print("2. Computing aggregations and generating plots...")
# Aggregation: Group by dataset, model, and batch index
agg_cols = list(METRIC_CONFIG.keys())
grouped = (
    master_df.groupby(["dataset", "model", "batch"])[agg_cols]
    .agg(["mean", "std"])
    .reset_index()
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
            means = model_data[(metric, "mean")].values
            stds = model_data[(metric, "std")].values

            # If only 1 run survived for a batch, std is NaN. Force to 0 for plotting continuity.
            stds = np.nan_to_num(stds, nan=0.0)

            # Boundary clipping
            lower_bound = means - stds
            upper_bound = means + stds

            # Log scale safety: matplotlib fill_between fails if lower_bound <= 0 on a log axis
            logical_min = config["min"]
            if config["y_scale"] == "log" and logical_min <= 0:
                logical_min = np.min(means)

            lower_bound = np.clip(lower_bound, logical_min, config["max"])
            upper_bound = np.clip(upper_bound, logical_min, config["max"])

            ax.plot(batches, means, label=model, color=color_map[model], linewidth=2.0)
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
