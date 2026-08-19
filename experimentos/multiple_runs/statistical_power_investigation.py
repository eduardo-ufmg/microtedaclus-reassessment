"""
statistical_power_investigation.py

Investigates the convergence of sample statistics toward their asymptotic
(full-sample) values as a function of the sample size k, without re-running
the underlying experiments. Two statistical pairs are tracked:

  - Location/spread pair 1: median and IQR (scale = reference IQR)
  - Location/spread pair 2: mean and standard deviation (scale = reference std)

Analysis is purely visual: the script computes the global (L_infinity,
worst-case across configurations) NMAE and NSE curves as a function of k
and plots them, with no smoothing or automatic elbow/optimality search.
The user inspects the plots directly.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

RESULTS_DIR = "./results"
FAMILIES = ["SCL2", "STR-B1", "STR-B2"]
DATASET_SIZES = [4000]
METRICS = ["ari", "total_time_ms", "memory_bytes", "microcluster_count"]

B = 10000  # Monte Carlo iterations
MIN_K = 3


def compute_convergence(data, k_values):
    N = len(data)

    ref_median = np.median(data)
    ref_iqr = np.percentile(data, 75) - np.percentile(data, 25)
    ref_mean = np.mean(data)
    ref_std = np.std(data, ddof=1)

    # Scale for the median/IQR pair
    scale_med_iqr = ref_iqr
    if np.isclose(scale_med_iqr, 0.0):
        scale_med_iqr = np.std(data, ddof=1)
        if np.isclose(scale_med_iqr, 0.0):
            scale_med_iqr = 1.0

    # Unified scale for the mean/std pair
    scale_mean_std = ref_iqr
    if np.isclose(scale_mean_std, 0.0):
        scale_mean_std = ref_std
        if np.isclose(scale_mean_std, 0.0):
            scale_mean_std = 1.0

    results = []
    rng = np.random.default_rng(seed=0)

    for k in tqdm(k_values, desc="Convergence by k", leave=False):
        rand_idx = rng.random((B, N)).argsort(axis=1)[:, :k]
        subsets = data[rand_idx]

        medians = np.median(subsets, axis=1)
        iqrs = np.percentile(subsets, 75, axis=1) - np.percentile(subsets, 25, axis=1)
        means = np.mean(subsets, axis=1)
        stds = np.std(subsets, axis=1, ddof=1)

        mae_med = np.mean(np.abs(medians - ref_median))
        se_med = np.std(medians, ddof=1) if B > 1 else 0.0
        nmae_med = mae_med / scale_med_iqr
        nse_med = se_med / scale_med_iqr

        mae_iqr = np.mean(np.abs(iqrs - ref_iqr))
        se_iqr = np.std(iqrs, ddof=1) if B > 1 else 0.0
        nmae_iqr = mae_iqr / scale_med_iqr
        nse_iqr = se_iqr / scale_med_iqr

        mae_mean = np.mean(np.abs(means - ref_mean))
        se_mean = np.std(means, ddof=1) if B > 1 else 0.0
        nmae_mean = mae_mean / scale_mean_std
        nse_mean = se_mean / scale_mean_std

        mae_std = np.mean(np.abs(stds - ref_std))
        se_std = np.std(stds, ddof=1) if B > 1 else 0.0
        nmae_std = mae_std / scale_mean_std
        nse_std = se_std / scale_mean_std

        results.append(
            {
                "k": k,
                "mae_med": mae_med,
                "se_med": se_med,
                "nmae_med": nmae_med,
                "nse_med": nse_med,
                "mae_iqr": mae_iqr,
                "se_iqr": se_iqr,
                "nmae_iqr": nmae_iqr,
                "nse_iqr": nse_iqr,
                "mae_mean": mae_mean,
                "se_mean": se_mean,
                "nmae_mean": nmae_mean,
                "nse_mean": nse_mean,
                "mae_std": mae_std,
                "se_std": se_std,
                "nmae_std": nmae_std,
                "nse_std": nse_std,
            }
        )

    return pd.DataFrame(results)


def _plot_stat_pair(ax, global_df, stat, title):
    """Plots Max NMAE / Max NSE for a single statistic (e.g. 'med', 'iqr',
    'mean', 'std') onto the given axis."""
    ax.plot(
        global_df.index,
        global_df[f"global_nmae_{stat}"],
        label="Max NMAE",
        color="blue",
    )
    ax.plot(
        global_df.index,
        global_df[f"global_nse_{stat}"],
        label="Max NSE",
        color="red",
        linestyle="--",
    )
    ax.set_title(title)
    ax.set_xlabel("Subset Size (k)")
    ax.set_ylabel("Normalized Error")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()


def plot_medianiqr_convergence(global_df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    _plot_stat_pair(axes[0], global_df, "med", "Median")
    _plot_stat_pair(axes[1], global_df, "iqr", "IQR")
    fig.suptitle("Global Worst-Case Convergence — Median / IQR")
    plt.tight_layout()

    out_path = os.path.join(RESULTS_DIR, "medianiqr_convergence.png")
    fig.savefig(out_path)
    plt.close(fig)


def plot_meanstd_convergence(global_df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    _plot_stat_pair(axes[0], global_df, "mean", "Mean")
    _plot_stat_pair(axes[1], global_df, "std", "Standard Deviation")
    fig.suptitle("Global Worst-Case Convergence — Mean / Std")
    plt.tight_layout()

    out_path = os.path.join(RESULTS_DIR, "meanstd_convergence.png")
    fig.savefig(out_path)
    plt.close(fig)


def plot_statistics_convergence(global_df):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    _plot_stat_pair(axes[0, 0], global_df, "med", "Median")
    _plot_stat_pair(axes[0, 1], global_df, "iqr", "IQR")
    _plot_stat_pair(axes[1, 0], global_df, "mean", "Mean")
    _plot_stat_pair(axes[1, 1], global_df, "std", "Standard Deviation")
    fig.suptitle("Global Worst-Case Convergence — All Statistics")
    plt.tight_layout()

    out_path = os.path.join(RESULTS_DIR, "statistics_convergence.png")
    fig.savefig(out_path)
    plt.close(fig)


def main():
    nse_med_dict, nse_iqr_dict = {}, {}
    nmae_med_dict, nmae_iqr_dict = {}, {}
    nse_mean_dict, nse_std_dict = {}, {}
    nmae_mean_dict, nmae_std_dict = {}, {}

    for family in tqdm(FAMILIES, desc="Families"):
        for n_total in tqdm(DATASET_SIZES, desc=f"{family} dataset sizes", leave=False):
            csv_path = os.path.join(RESULTS_DIR, f"{family}_{n_total}.csv")
            if not os.path.exists(csv_path):
                continue

            raw_df = pd.read_csv(csv_path, index_col=0)
            if "status" in raw_df.columns:
                raw_df = raw_df[raw_df["status"] == "OK"]

            N = len(raw_df)
            if N <= MIN_K:
                continue

            k_values = np.arange(MIN_K, N)

            for metric in tqdm(
                METRICS, desc=f"{family}_{n_total} metrics", leave=False
            ):
                if metric not in raw_df.columns:
                    continue

                data = raw_df[metric].dropna().to_numpy()
                if len(data) <= MIN_K:
                    continue

                conv_df = compute_convergence(data, k_values)
                conv_df.set_index("k", inplace=True)

                config_key = f"{family}_{n_total}_{metric}"
                nse_med_dict[config_key] = conv_df["nse_med"]
                nse_iqr_dict[config_key] = conv_df["nse_iqr"]
                nmae_med_dict[config_key] = conv_df["nmae_med"]
                nmae_iqr_dict[config_key] = conv_df["nmae_iqr"]

                nse_mean_dict[config_key] = conv_df["nse_mean"]
                nse_std_dict[config_key] = conv_df["nse_std"]
                nmae_mean_dict[config_key] = conv_df["nmae_mean"]
                nmae_std_dict[config_key] = conv_df["nmae_std"]

    # Global L_inf aggregation (worst case across all configurations)
    global_df = pd.DataFrame(
        {
            "global_nse_med": pd.DataFrame(nse_med_dict).max(axis=1),
            "global_nse_iqr": pd.DataFrame(nse_iqr_dict).max(axis=1),
            "global_nmae_med": pd.DataFrame(nmae_med_dict).max(axis=1),
            "global_nmae_iqr": pd.DataFrame(nmae_iqr_dict).max(axis=1),
            "global_nse_mean": pd.DataFrame(nse_mean_dict).max(axis=1),
            "global_nse_std": pd.DataFrame(nse_std_dict).max(axis=1),
            "global_nmae_mean": pd.DataFrame(nmae_mean_dict).max(axis=1),
            "global_nmae_std": pd.DataFrame(nmae_std_dict).max(axis=1),
        }
    ).dropna()

    plot_medianiqr_convergence(global_df)
    plot_meanstd_convergence(global_df)
    plot_statistics_convergence(global_df)

    print("================== GLOBAL CONVERGENCE PLOTS ==================")
    print(f"Saved: {os.path.join(RESULTS_DIR, 'medianiqr_convergence.png')}")
    print(f"Saved: {os.path.join(RESULTS_DIR, 'meanstd_convergence.png')}")
    print(f"Saved: {os.path.join(RESULTS_DIR, 'statistics_convergence.png')}")
    print("================================================================\n")


if __name__ == "__main__":
    main()
