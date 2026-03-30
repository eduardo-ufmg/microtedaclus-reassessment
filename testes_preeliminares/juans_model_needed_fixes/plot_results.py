import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_results(csv_path: Path, output_dir: Path, ref_csv_path: Path):
    df = pd.read_csv(csv_path)
    ref_df = pd.read_csv(ref_csv_path)

    if df.empty:
        raise ValueError(f"CSV file {csv_path} is empty")
    if ref_df.empty:
        raise ValueError(f"Reference CSV file {ref_csv_path} is empty")

    # Ensure numeric conversion
    for col in [
        "batch_size",
        "tune_ari_chunk",
        "tune_fit_lat_sec",
        "tune_pred_lat_sec",
        "test_ari_chunk",
        "test_fit_lat_sec",
        "test_pred_lat_sec",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in [
        "tune_ari_global",
        "test_ari_global",
        "test_fit_lat_sec",
        "test_pred_lat_sec",
    ]:
        if col in ref_df.columns:
            ref_df[col] = pd.to_numeric(ref_df[col], errors="coerce")

    datasets = sorted(df["dataset"].unique())
    n = len(datasets)
    if n == 0:
        raise ValueError("No dataset values found in CSV")

    # Plot 1: times (fit and predict) by dataset
    cols = 4
    rows = (n + cols - 1) // cols
    fig1, axs1 = plt.subplots(
        rows, cols, figsize=(6 * cols, 5 * rows), squeeze=False, sharey=True
    )
    for i, dataset in enumerate(datasets):
        r = i // cols
        c = i % cols
        sub = axs1[r][c]
        ds = df[df["dataset"] == dataset].sort_values("batch_size")

        if "test_fit_lat_sec" in ds.columns:
            sub.plot(
                ds["batch_size"],
                ds["test_fit_lat_sec"],
                marker="o",
                label="test_fit_lat_sec",
            )
        if "test_pred_lat_sec" in ds.columns:
            sub.plot(
                ds["batch_size"],
                ds["test_pred_lat_sec"],
                marker="o",
                label="test_pred_lat_sec",
            )

        ref_row = ref_df[ref_df["dataset"] == dataset]
        if not ref_row.empty:
            if "test_fit_lat_sec" in ref_row.columns and pd.notna(
                ref_row["test_fit_lat_sec"].iloc[0]
            ):
                sub.axhline(
                    ref_row["test_fit_lat_sec"].iloc[0],
                    color="red",
                    linestyle="--",
                    label="ref_test_fit_lat_sec",
                )
            if "test_pred_lat_sec" in ref_row.columns and pd.notna(
                ref_row["test_pred_lat_sec"].iloc[0]
            ):
                sub.axhline(
                    ref_row["test_pred_lat_sec"].iloc[0],
                    color="purple",
                    linestyle="--",
                    label="ref_test_pred_lat_sec",
                )

        sub.set_title(f"{dataset} - times")
        sub.set_xlabel("batch_size")
        sub.set_ylabel("time (s)")
        sub.set_xscale("log", base=2)
        sub.set_yscale("log")
        sub.legend()
        sub.grid(True, which="both", linestyle="--", alpha=0.5)

    fig1.suptitle("Fit and Predict Time by Batch Size")
    fig1.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig1_path = output_dir / "plot_time.png"
    fig1.savefig(fig1_path)
    print(f"Saved times figure to: {fig1_path}")

    # Plot 2: ARI scores by dataset
    cols = 4
    rows = (n + cols - 1) // cols
    fig2, axs2 = plt.subplots(
        rows, cols, figsize=(6 * cols, 5 * rows), squeeze=False, sharey=True
    )
    for i, dataset in enumerate(datasets):
        r = i // cols
        c = i % cols
        sub = axs2[r][c]
        ds = df[df["dataset"] == dataset].sort_values("batch_size")

        if "test_ari_chunk" in ds.columns:
            sub.plot(
                ds["batch_size"],
                ds["test_ari_chunk"],
                marker="o",
                label="test_ari_chunk",
            )
        if "tune_ari_chunk" in ds.columns:
            sub.plot(
                ds["batch_size"],
                ds["tune_ari_chunk"],
                marker="o",
                label="tune_ari_chunk",
            )

        ref_row = ref_df[ref_df["dataset"] == dataset]
        if (
            not ref_row.empty
            and "test_ari_global" in ref_row.columns
            and pd.notna(ref_row["test_ari_global"].iloc[0])
        ):
            sub.axhline(
                ref_row["test_ari_global"].iloc[0],
                color="green",
                linestyle="--",
                label="ref_test_ari_global",
            )

        sub.set_title(f"{dataset} - ARI")
        sub.set_xlabel("batch_size")
        sub.set_ylabel("ARI")
        sub.set_xscale("log", base=2)
        sub.legend()
        sub.grid(True, which="both", linestyle="--", alpha=0.5)

    fig2.suptitle("ARI scores by Batch Size")
    # Hide unused axes
    for j in range(n, rows * cols):
        r = j // cols
        c = j % cols
        axs2[r][c].axis("off")

    fig2.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig2_path = output_dir / "plot_ari.png"
    fig2.savefig(fig2_path)
    print(f"Saved ARI figure to: {fig2_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot predict-then-fit results CSV")
    parser.add_argument(
        "--csv", type=Path, default="predictthenfit_results.csv", help="Input CSV path"
    )
    parser.add_argument(
        "--ref",
        type=Path,
        default="fitthenpredict_results.csv",
        help="Reference CSV path for horizontal lines",
    )
    parser.add_argument(
        "--out", type=Path, default=".", help="Output directory for figures"
    )
    args = parser.parse_args()

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_results(args.csv, out_dir, args.ref)
