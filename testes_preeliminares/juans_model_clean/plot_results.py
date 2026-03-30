import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def read_predictthenfit(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"CSV file {csv_path} is empty")

    expected_columns = [
        "dataset",
        "batch_size",
        "test_ari_chunk",
        "test_fit_lat_sec",
        "test_pred_lat_sec",
    ]

    for col in expected_columns:
        if col not in df.columns:
            raise ValueError(f"Missing expected column '{col}' in {csv_path}")

    df["batch_size"] = pd.to_numeric(df["batch_size"], errors="coerce")
    for col in ["test_ari_chunk", "test_fit_lat_sec", "test_pred_lat_sec"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def plot_all(primary_df: pd.DataFrame, compare_df: pd.DataFrame, output_dir: Path):
    datasets = sorted(set(primary_df["dataset"]).union(set(compare_df["dataset"])))
    n = len(datasets)
    if n == 0:
        raise ValueError("No dataset values found for plotting")

    # Latency comparison: fit and predict, and test
    fig_lat, axs_lat = plt.subplots(
        1, n, figsize=(6 * n, 5), squeeze=False, sharey=True
    )
    for i, ds_name in enumerate(datasets):
        ax = axs_lat[0][i]

        p_df = primary_df[primary_df["dataset"] == ds_name].sort_values("batch_size")
        c_df = compare_df[compare_df["dataset"] == ds_name].sort_values("batch_size")

        if not p_df.empty:
            ax.plot(
                p_df["batch_size"],
                p_df["test_fit_lat_sec"],
                marker="o",
                label="primary_test_fit",
            )
            ax.plot(
                p_df["batch_size"],
                p_df["test_pred_lat_sec"],
                marker="o",
                label="primary_test_pred",
            )
        if not c_df.empty:
            ax.plot(
                c_df["batch_size"],
                c_df["test_fit_lat_sec"],
                marker="s",
                linestyle="--",
                label="compare_test_fit",
            )
            ax.plot(
                c_df["batch_size"],
                c_df["test_pred_lat_sec"],
                marker="s",
                linestyle="--",
                label="compare_test_pred",
            )

        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_title(f"{ds_name} - latency")
        ax.set_xlabel("batch_size")
        ax.set_ylabel("seconds")
        ax.legend()
        ax.grid(True, which="both", linestyle="--", alpha=0.5)

    fig_lat.suptitle("Compare test fit/pred latency")
    fig_lat.tight_layout(rect=(0, 0.03, 1, 0.95))
    lat_path = output_dir / "compare_latency.png"
    fig_lat.savefig(lat_path, dpi=150)
    print(f"Saved latency comparison plot to: {lat_path}")

    # ARI comparison
    fig_ari, axs_ari = plt.subplots(
        1, n, figsize=(6 * n, 5), squeeze=False, sharey=True
    )
    for i, ds_name in enumerate(datasets):
        ax = axs_ari[0][i]

        p_df = primary_df[primary_df["dataset"] == ds_name].sort_values("batch_size")
        c_df = compare_df[compare_df["dataset"] == ds_name].sort_values("batch_size")

        if not p_df.empty:
            ax.plot(
                p_df["batch_size"],
                p_df["test_ari_chunk"],
                marker="o",
                label="primary_test_ari",
            )
        if not c_df.empty:
            ax.plot(
                c_df["batch_size"],
                c_df["test_ari_chunk"],
                marker="s",
                label="compare_test_ari",
            )

        ax.set_xscale("log", base=2)
        ax.set_title(f"{ds_name} - ARI")
        ax.set_xlabel("batch_size")
        ax.set_ylabel("ARI")
        ax.legend()
        ax.grid(True, which="both", linestyle="--", alpha=0.5)

    fig_ari.suptitle("Compare test ARI")
    fig_ari.tight_layout(rect=(0, 0.03, 1, 0.95))
    ari_path = output_dir / "compare_ari.png"
    fig_ari.savefig(ari_path, dpi=150)
    print(f"Saved ARI comparison plot to: {ari_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot predictthenfit CSV latency and ARI comparison"
    )
    parser.add_argument(
        "--primary",
        type=Path,
        default="predictthenfit_results.csv",
        help="Primary CSV path (in current folder)",
    )
    parser.add_argument(
        "--compare",
        type=Path,
        default="../juans_model_needed_fixes/predictthenfit_results.csv",
        help="Comparison CSV path",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=".",
        help="Output directory for figures",
    )
    args = parser.parse_args()

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    primary_df = read_predictthenfit(args.primary)
    compare_df = read_predictthenfit(args.compare)
    plot_all(primary_df, compare_df, out_dir)
