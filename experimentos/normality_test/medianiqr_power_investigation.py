"""
medianiqr_power_investigation.py

Investigates the convergence of the sample median and IQR toward their
asymptotic values as a function of the sample size k, without re-running
the experiments. Computes both individual and L_infinity global optimums.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = "./results"
FAMILIES = ["SCL2", "STR-B1", "STR-B2"]
DATASET_SIZES = [4000, 8000]
METRICS = ["ari", "total_time_ms", "memory_bytes", "microcluster_count"]

B = 1000  # Monte Carlo iterations
MIN_K = 3
EPSILON = 0.01  # Tolerance for the discrete derivative of NSE


def compute_convergence(data, k_values):
    N = len(data)
    ref_median = np.median(data)
    ref_iqr = np.percentile(data, 75) - np.percentile(data, 25)

    scale = ref_iqr
    if np.isclose(scale, 0.0):
        scale = np.std(data)
        if np.isclose(scale, 0.0):
            scale = 1.0

    results = []
    rng = np.random.default_rng(seed=0)

    for k in k_values:
        rand_idx = rng.random((B, N)).argsort(axis=1)[:, :k]
        subsets = data[rand_idx]

        medians = np.median(subsets, axis=1)
        iqrs = np.percentile(subsets, 75, axis=1) - np.percentile(subsets, 25, axis=1)

        mae_med = np.mean(np.abs(medians - ref_median))
        se_med = np.std(medians, ddof=1) if B > 1 else 0.0
        nmae_med = mae_med / scale
        nse_med = se_med / scale

        mae_iqr = np.mean(np.abs(iqrs - ref_iqr))
        se_iqr = np.std(iqrs, ddof=1) if B > 1 else 0.0
        nmae_iqr = mae_iqr / scale
        nse_iqr = se_iqr / scale

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
            }
        )

    return pd.DataFrame(results)


def find_elbow(series, epsilon):
    diffs = np.abs(np.diff(series))
    stable_indices = np.where(diffs < epsilon)[0]

    if len(stable_indices) == 0:
        return series.index.max()

    return series.index[stable_indices[0] + 1]


def plot_global_convergence(global_df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(
        global_df.index, global_df["global_nmae_med"], label="Max NMAE", color="blue"
    )
    axes[0].plot(
        global_df.index,
        global_df["global_nse_med"],
        label="Max NSE",
        color="red",
        linestyle="--",
    )
    axes[0].set_title("Global L_inf Median Convergence")
    axes[0].set_xlabel("Subset Size (k)")
    axes[0].set_ylabel("Normalized Error")
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend()

    axes[1].plot(
        global_df.index, global_df["global_nmae_iqr"], label="Max NMAE", color="blue"
    )
    axes[1].plot(
        global_df.index,
        global_df["global_nse_iqr"],
        label="Max NSE",
        color="red",
        linestyle="--",
    )
    axes[1].set_title("Global L_inf IQR Convergence")
    axes[1].set_xlabel("Subset Size (k)")
    axes[1].set_ylabel("Normalized Error")
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend()

    fig.suptitle(f"Global Worst-Case Convergence Across All Configurations")
    plt.tight_layout()

    out_path = os.path.join(RESULTS_DIR, "medianiqr_convergence_GLOBAL.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main():
    summary_records = []

    nse_med_dict = {}
    nse_iqr_dict = {}
    nmae_med_dict = {}
    nmae_iqr_dict = {}

    for family in FAMILIES:
        for n_total in DATASET_SIZES:
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

            for metric in METRICS:
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

                opt_k_med = find_elbow(conv_df["nse_med"], EPSILON)
                opt_k_iqr = find_elbow(conv_df["nse_iqr"], EPSILON)

                summary_records.append(
                    {
                        "Family": family,
                        "N_Total": n_total,
                        "Metric": metric,
                        "Total_Runs": len(data),
                        f"Opt_k_Med": opt_k_med,
                        f"Opt_k_IQR": opt_k_iqr,
                    }
                )

    summary_df = pd.DataFrame(summary_records)

    # Global L_inf optimization
    df_nse_med = pd.DataFrame(nse_med_dict)
    df_nse_iqr = pd.DataFrame(nse_iqr_dict)
    df_nmae_med = pd.DataFrame(nmae_med_dict)
    df_nmae_iqr = pd.DataFrame(nmae_iqr_dict)

    global_df = pd.DataFrame(
        {
            "global_nse_med": df_nse_med.max(axis=1),
            "global_nse_iqr": df_nse_iqr.max(axis=1),
            "global_nmae_med": df_nmae_med.max(axis=1),
            "global_nmae_iqr": df_nmae_iqr.max(axis=1),
        }
    ).dropna()

    global_opt_med = find_elbow(global_df["global_nse_med"], EPSILON)
    global_opt_iqr = find_elbow(global_df["global_nse_iqr"], EPSILON)

    plot_global_convergence(global_df)

    print("\n================= INDIVIDUAL K SUMMARY ==================")
    print(summary_df.to_string(index=False))
    print("=========================================================\n")

    print("================== GLOBAL L_INF SUMMARY ==================")
    print(f"Global Optimal k (Median, eps={EPSILON}): {global_opt_med}")
    print(f"Global Optimal k (IQR, eps={EPSILON}):    {global_opt_iqr}")
    print(
        f"Recommended Suite k:                     {max(global_opt_med, global_opt_iqr)}"
    )
    print("=========================================================\n")

    out_csv = os.path.join(RESULTS_DIR, "medianiqr_optimal_k.csv")
    summary_df.to_csv(out_csv, index=False)


if __name__ == "__main__":
    main()
