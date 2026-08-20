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


def last_stored_k(path):
    if not os.path.exists(path):
        return MIN_K - 1
    try:
        existing = pd.read_csv(path, index_col=0)
        if existing.empty:
            return MIN_K - 1
        return int(existing.index.max())
    except (pd.errors.EmptyDataError, ValueError):
        return MIN_K - 1


def append_conv_row(path, k, row_dict):
    header = not os.path.exists(path)
    row_df = pd.DataFrame([row_dict], index=[k])
    row_df.index.name = "k"
    row_df.to_csv(path, mode="a", header=header)


def compute_convergence_incrementally(data, k_values, out_path):
    N = len(data)

    ref_median = np.median(data)
    ref_iqr = np.percentile(data, 75) - np.percentile(data, 25)

    scale_med_iqr = ref_iqr
    if np.isclose(scale_med_iqr, 0.0):
        scale_med_iqr = np.std(data, ddof=1)
        if np.isclose(scale_med_iqr, 0.0):
            scale_med_iqr = 1.0

    last_k = last_stored_k(out_path)
    k_to_compute = [k for k in k_values if k > last_k]

    for k in tqdm(k_to_compute, desc="Convergence by k", leave=False):
        rng = np.random.default_rng(seed=k)
        rand_idx = rng.random((B, N)).argsort(axis=1)[:, :k]
        subsets = data[rand_idx]

        medians = np.median(subsets, axis=1)
        iqrs = np.percentile(subsets, 75, axis=1) - np.percentile(subsets, 25, axis=1)

        mae_med = np.mean(np.abs(medians - ref_median))
        se_med = np.std(medians, ddof=1) if B > 1 else 0.0
        nmae_med = mae_med / scale_med_iqr
        nse_med = se_med / scale_med_iqr

        mae_iqr = np.mean(np.abs(iqrs - ref_iqr))
        se_iqr = np.std(iqrs, ddof=1) if B > 1 else 0.0
        nmae_iqr = mae_iqr / scale_med_iqr
        nse_iqr = se_iqr / scale_med_iqr

        row_dict = {
            "mae_med": mae_med,
            "se_med": se_med,
            "nmae_med": nmae_med,
            "nse_med": nse_med,
            "mae_iqr": mae_iqr,
            "se_iqr": se_iqr,
            "nmae_iqr": nmae_iqr,
            "nse_iqr": nse_iqr,
        }
        append_conv_row(out_path, k, row_dict)

    return pd.read_csv(out_path, index_col=0)


def _plot_stat_pair(ax, global_df, stat, title):
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


def analyze_max_nmae_critical_points(global_df):
    k = global_df.index.to_numpy(dtype=float)
    y = global_df["global_nmae_med"].to_numpy(dtype=float)

    dy = np.gradient(y, k)
    ddy = np.gradient(dy, k)

    x1, y1 = k[0], y[0]
    x2, y2 = k[-1], y[-1]

    num = np.abs((x2 - x1) * (y - y1) - (y2 - y1) * (k - x1))
    den = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    dist_secant = num / den
    ddist_secant = np.gradient(dist_secant, k)

    signals = {
        "1st Derivative": dy,
        "2nd Derivative": ddy,
        "Secant Line Distance": dist_secant,
        "Secant Distance Derivative": ddist_secant,
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, (title, sig) in zip(axes, signals.items()):
        ax.plot(k, sig, label=title, color="black")

        idx_max = np.nanargmax(sig)
        idx_min = np.nanargmin(sig)

        ax.scatter(
            k[idx_max],
            sig[idx_max],
            color="red",
            label=f"Global Max (k={k[idx_max]:.0f})",
            zorder=5,
        )
        ax.scatter(
            k[idx_min],
            sig[idx_min],
            color="blue",
            label=f"Global Min (k={k[idx_min]:.0f})",
            zorder=5,
        )

        ax.set_title(title)
        ax.set_xlabel("Subset Size (k)")
        ax.set_ylabel("Signal Amplitude")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()

    fig.suptitle("Max NMAE (Median) Critical Point Analysis (Original Domains)")
    plt.tight_layout()

    out_path = os.path.join(RESULTS_DIR, "max_nmae_critical_points.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    nse_med_dict, nse_iqr_dict = {}, {}
    nmae_med_dict, nmae_iqr_dict = {}, {}

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

                out_conv_path = os.path.join(
                    RESULTS_DIR, f"conv_{family}_{n_total}_{metric}.csv"
                )
                conv_df = compute_convergence_incrementally(
                    data, k_values, out_conv_path
                )

                config_key = f"{family}_{n_total}_{metric}"
                nse_med_dict[config_key] = conv_df["nse_med"]
                nse_iqr_dict[config_key] = conv_df["nse_iqr"]
                nmae_med_dict[config_key] = conv_df["nmae_med"]
                nmae_iqr_dict[config_key] = conv_df["nmae_iqr"]

    global_df = pd.DataFrame(
        {
            "global_nse_med": pd.DataFrame(nse_med_dict).max(axis=1),
            "global_nse_iqr": pd.DataFrame(nse_iqr_dict).max(axis=1),
            "global_nmae_med": pd.DataFrame(nmae_med_dict).max(axis=1),
            "global_nmae_iqr": pd.DataFrame(nmae_iqr_dict).max(axis=1),
        }
    ).dropna()

    plot_medianiqr_convergence(global_df)
    analyze_max_nmae_critical_points(global_df)

    print("================== GLOBAL CONVERGENCE PLOTS ==================")
    print(f"Saved: {os.path.join(RESULTS_DIR, 'medianiqr_convergence.png')}")
    print("================================================================\n")


if __name__ == "__main__":
    main()
