import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def generate_visualization(csv_file: str, output_image: str) -> None:
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: {csv_file} not found.")
        sys.exit(1)

    canonical_envs = ["Java", "Bridge", "River"]
    active_envs = [env for env in canonical_envs if env in df["Environment"].unique()]
    algorithms = df["Algorithm"].unique()

    n_envs = len(active_envs)
    x = np.arange(len(algorithms))
    bar_width = 0.8 / n_envs
    offsets = np.linspace(-0.4 + bar_width / 2, 0.4 - bar_width / 2, n_envs)

    colors = cm.magma(np.linspace(0.2, 0.8, n_envs))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for i, env in enumerate(active_envs):
        env_data = (
            df[df["Environment"] == env].set_index("Algorithm").reindex(algorithms)
        )

        # 1. Execution Time Plot (Log Scale with Asymmetric Bounds)
        time_mean = env_data["OS_Wall_Clock_sec_mean"].values
        time_std = env_data["OS_Wall_Clock_sec_std"].fillna(0).values

        lower_time_err = np.clip(time_std, 0, time_mean - 1e-4)
        upper_time_err = time_std

        axes[0].bar(
            x + offsets[i],
            time_mean,
            bar_width,
            yerr=[lower_time_err, upper_time_err],
            label=env,
            color=colors[i],
            capsize=5,
            edgecolor="black",
        )

        # 2. Memory Footprint Plot (Linear Scale)
        mem_mean = env_data["OS_Peak_RSS_MB_mean"].values
        mem_std = env_data["OS_Peak_RSS_MB_std"].fillna(0).values

        axes[1].bar(
            x + offsets[i],
            mem_mean,
            bar_width,
            yerr=mem_std,
            label=env,
            color=colors[i],
            capsize=5,
            edgecolor="black",
        )

    axes[0].set_title("Execution Time (OS Wall Clock)", fontsize=12)
    axes[0].set_ylabel("Time (seconds)", fontsize=10)
    axes[0].set_xlabel("Algorithm", fontsize=10)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(algorithms)
    axes[0].set_yscale("log")
    axes[0].grid(True, which="both", ls="--", alpha=0.5)

    axes[1].set_title("Memory Footprint (OS Peak RSS)", fontsize=12)
    axes[1].set_ylabel("Memory (MB)", fontsize=10)
    axes[1].set_xlabel("Algorithm", fontsize=10)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(algorithms)
    axes[1].grid(True, axis="y", ls="--", alpha=0.5)

    for ax in axes:
        ax.legend(title="Environment")

    plt.tight_layout()
    plt.savefig(output_image, format="png")
    print(f"Visualization successfully saved to {output_image}")


if __name__ == "__main__":
    input_csv = sys.argv[1] if len(sys.argv) > 1 else "benchmark_results.csv"
    output_fig = sys.argv[2] if len(sys.argv) > 2 else "benchmark_analysis.png"
    generate_visualization(input_csv, output_fig)
