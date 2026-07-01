import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def generate_visualization(csv_file: str, output_image: str) -> None:
    # 1. Load Data
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: {csv_file} not found.")
        sys.exit(1)

    # 2. Data Cleansing
    # Coerce the nonsensical "<class 'float'>" artifacts in Java's internal memory to NaN
    df["Internal_Peak_RSS_MB"] = pd.to_numeric(
        df["Internal_Peak_RSS_MB"], errors="coerce"
    )

    # 3. Future-Proofing
    # Define canonical order for environments to maintain consistent coloring when 'River' is added
    canonical_envs = ["Java", "Bridge", "River"]
    active_envs = [env for env in canonical_envs if env in df["Environment"].unique()]

    # 4. Plot Setup
    sns.set_theme(style="whitegrid", context="paper")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 5. Execution Time Plot (OS Wall Clock)
    sns.barplot(
        data=df,
        x="Algorithm",
        y="OS_Wall_Clock_sec",
        hue="Environment",
        hue_order=active_envs,
        ax=axes[0],
        palette="magma",
    )
    axes[0].set_title(
        "Execution Time Comparison (OS Wall Clock)", fontsize=12, fontweight="bold"
    )
    axes[0].set_ylabel("Time (seconds)", fontsize=10)
    axes[0].set_xlabel("Algorithm", fontsize=10)
    axes[0].set_yscale("log")  # Log scale mitigates the magnitude distortion

    # 6. Memory Footprint Plot (OS Peak RSS)
    sns.barplot(
        data=df,
        x="Algorithm",
        y="OS_Peak_RSS_MB",
        hue="Environment",
        hue_order=active_envs,
        ax=axes[1],
        palette="magma",
    )
    axes[1].set_title(
        "Memory Footprint Comparison (OS Peak RSS)", fontsize=12, fontweight="bold"
    )
    axes[1].set_ylabel("Memory (MB)", fontsize=10)
    axes[1].set_xlabel("Algorithm", fontsize=10)

    # 7. Final Formatting
    for ax in axes:
        ax.legend(title="Environment")

    plt.tight_layout()
    plt.savefig(output_image, dpi=300, format="png")
    print(f"Visualization successfully saved to {output_image}")


if __name__ == "__main__":
    input_csv = sys.argv[1] if len(sys.argv) > 1 else "benchmark_results.csv"
    output_fig = sys.argv[2] if len(sys.argv) > 2 else "benchmark_analysis.png"
    generate_visualization(input_csv, output_fig)
