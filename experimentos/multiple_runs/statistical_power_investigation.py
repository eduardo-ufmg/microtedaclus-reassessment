import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
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
        nmae_med = mae_med / scale_med_iqr

        mae_iqr = np.mean(np.abs(iqrs - ref_iqr))
        nmae_iqr = mae_iqr / scale_med_iqr

        row_dict = {
            "mae_med": mae_med,
            "nmae_med": nmae_med,
            "mae_iqr": mae_iqr,
            "nmae_iqr": nmae_iqr,
        }
        append_conv_row(out_path, k, row_dict)

    return pd.read_csv(out_path, index_col=0)


def _plot_stat(ax, global_df, stat, title, y_iso=None):
    k = global_df.index.to_numpy(dtype=float)
    y = global_df[f"global_nmae_{stat}"].to_numpy(dtype=float)

    k_max = k.max()
    y_max = y.max()

    ax.set_xlabel("Subset Size (k)")
    ax.set_ylabel("Max NMAE")

    ax.plot(k, y, label="Max NMAE", color="blue", alpha=0.4)

    ax.set_title(title)

    ax.set_xlim(left=-50, right=1050)
    ax.set_ylim(bottom=-0.05)
    ax.set_aspect(k_max / y_max, adjustable="box")

    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()


def plot_medianiqr_convergence_with_k(global_df, k_star, y_iso_med=None):
    fig, axes = plt.subplots(
        1,
        2,
        constrained_layout=True,
    )

    _plot_stat(axes[0], global_df, "med", "Median", y_iso=y_iso_med)
    axes[0].axvline(k_star, color="black", linestyle=":", label=f"k*={k_star:.0f}")
    axes[0].set_xticks(np.append(axes[0].get_xticks(), k_star))
    axes[0].legend()

    _plot_stat(axes[1], global_df, "iqr", "IQR")
    axes[1].axvline(k_star, color="black", linestyle=":", label=f"k*={k_star:.0f}")
    axes[1].set_xticks(np.append(axes[1].get_xticks(), k_star))
    axes[1].legend()

    for ax in axes:
        ax.set_xlim(left=-50, right=1050)
        ax.set_ylim(bottom=-0.05)
        ax.grid(True, linestyle="--", alpha=0.6)

    fig.suptitle(
        f"Global Worst-Case Convergence",
        y=0.875,
    )

    out_path = os.path.join(RESULTS_DIR, "medianiqr_convergence_with_k.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def compute_isotonic_elbow(k_array, y_array):
    ir = IsotonicRegression(increasing=False, out_of_bounds="clip")
    y_iso = ir.fit_transform(k_array, y_array)

    k_min, k_max = k_array.min(), k_array.max()
    y_min, y_max = y_iso.min(), y_iso.max()

    if np.isclose(k_max, k_min) or np.isclose(y_max, y_min):
        return k_min, y_iso

    k_norm = (k_array - k_min) / (k_max - k_min)
    y_norm = (y_iso - y_min) / (y_max - y_min)

    x1, y1 = k_norm[0], y_norm[0]
    x2, y2 = k_norm[-1], y_norm[-1]

    num = np.abs((x2 - x1) * (y_norm - y1) - (y2 - y1) * (k_norm - x1))
    den = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    dist_secant = num / den

    idx_star = np.nanargmax(dist_secant)
    k_star = k_array[idx_star]

    return k_star, y_iso


def main():
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
                nmae_med_dict[config_key] = conv_df["nmae_med"]
                nmae_iqr_dict[config_key] = conv_df["nmae_iqr"]

    if not nmae_med_dict or not nmae_iqr_dict:
        print("No valid convergence data generated. Exiting.")
        return

    global_df = pd.DataFrame(
        {
            "global_nmae_med": pd.DataFrame(nmae_med_dict).max(axis=1),
            "global_nmae_iqr": pd.DataFrame(nmae_iqr_dict).max(axis=1),
        }
    ).dropna()

    k_array = global_df.index.to_numpy(dtype=float)
    y_med_array = global_df["global_nmae_med"].to_numpy(dtype=float)

    k_star, y_iso_med = compute_isotonic_elbow(k_array, y_med_array)

    print("\n================== GLOBAL CONVERGENCE PLOTS ==================")
    plot_medianiqr_convergence_with_k(global_df, k_star, y_iso_med=y_iso_med)
    print("================================================================\n")


if __name__ == "__main__":
    main()
