"""Generate dataset visualizations for the thesis figures.

The script writes all outputs to a local ``figures/`` subdirectory next to
this file.
"""

import logging
import os
import sys
import urllib.request
from io import StringIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rpy2.rinterface_lib.callbacks
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from scipy.io import arff
from sklearn.datasets import make_blobs

rpy2.rinterface_lib.callbacks.logger.setLevel(logging.ERROR)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

RNG_SEED = 0
BATCH_SIZE = 100
SCALABILITY_DIM = 2
LABEL_PALETTE = plt.get_cmap("tab20", 20)

CASSINI_BATCH_INDEX = 2
CONCEPT_DRIFT_SPECS = [
    ("StreamBenchmark(1)", "Benchmark(1)", 4000),
    ("StreamBenchmark(2)", "Benchmark(2)", 4000),
    ("RandomRBFGeneratorEvents", "RandomRBFGeneratorEvents", 10000),
]

try:
    stream = importr("stream")
    stream_moa = importr("streamMOA")
    mlbench = importr("mlbench")
except Exception as exc:
    print(f"Critical Error loading R dependencies via rpy2: {exc}")
    sys.exit(1)


def get_chameleon(url):
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as response:
        data, _ = arff.loadarff(StringIO(response.read().decode("utf-8")))

    df = pd.DataFrame(data)
    target_col = df.columns[-1]
    y_raw = df[target_col].str.decode("utf-8")
    y_raw = y_raw.replace("noise", "-1")
    y = y_raw.astype(int).values
    X = df.drop(columns=[target_col]).values
    return X, y


def load_cassini(shuffle=False, seed=RNG_SEED):
    robjects.r("data <- mlbench.cassini(4000)")
    X = np.array(robjects.r("data$x"))
    y = np.array(robjects.r("as.numeric(data$classes)"))
    y = np.nan_to_num(y, nan=-1).astype(int)

    if shuffle:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(X))
        return X[idx], y[idx]

    return X, y


def load_r_stream(name):
    if name == "RandomRBFGeneratorEvents":
        robjects.r("""
			dsd <- DSD_RandomRBFGeneratorEvents(
				eventFrequency = 1000L,
				eventMergeSplitOption = TRUE,
				eventDeleteCreate = TRUE
			)
			data <- get_points(dsd, n = 10000, info = TRUE)
			""")
    elif name == "Benchmark(1)":
        robjects.r("dsd <- DSD_Benchmark(1)")
        robjects.r("data <- get_points(dsd, n = 4000, info = TRUE)")
    elif name == "Benchmark(2)":
        robjects.r("dsd <- DSD_Benchmark(2)")
        robjects.r("data <- get_points(dsd, n = 4000, info = TRUE)")
    else:
        raise ValueError(f"Unknown R dataset: {name}")

    X = np.array(robjects.r("as.matrix(data[, -ncol(data)])"))
    y = np.array(robjects.r("as.numeric(data[, ncol(data)])"))
    y = np.nan_to_num(y, nan=-1).astype(int)
    return X, y


def generate_scl(dim, with_noise=True, seed=RNG_SEED):
    X, y = make_blobs(
        n_samples=4000,
        centers=3,
        n_features=dim,
        cluster_std=0.5,
        random_state=seed,
    )[0:2]

    if not with_noise:
        return X, y

    rng = np.random.default_rng(seed)
    noise = rng.uniform(low=np.min(X, axis=0), high=np.max(X, axis=0), size=(400, dim))
    X_full = np.vstack([X, noise])
    y_full = np.concatenate([y, np.full(400, -1)])
    idx = rng.permutation(len(X_full))
    return X_full[idx], y_full[idx]


def first_batch(X, y, batch_index=0, batch_size=BATCH_SIZE):
    start_idx = batch_index * batch_size
    end_idx = start_idx + batch_size
    return X[start_idx:end_idx], y[start_idx:end_idx]


def label_colors(y):
    labels = np.asarray(y).astype(int)
    color_map: dict[int, str | tuple[float, float, float, float]] = {-1: "#202020"}

    for label in np.unique(labels):
        if label != -1:
            # Keep the RGBA tuple returned by the colormap. Converting it to
            # a string makes newer Matplotlib versions reject the color.
            color_map[int(label)] = LABEL_PALETTE((int(label) % 20) / 19)

    return [color_map[int(label)] for label in labels]


def plot_scatter(ax, X, y, title, xlim=None, ylim=None, point_size=7, alpha=0.85):
    if len(X) == 0:
        ax.set_title(f"{title} (empty)")
        ax.axis("off")
        return

    colors = label_colors(y)
    ax.scatter(X[:, 0], X[:, 1], c=colors, s=point_size, alpha=alpha, linewidths=0)
    ax.set_title(title)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.grid(True, linestyle="--", alpha=0.3)


def plot_cassini_comparison():
    X_ordered, y_ordered = load_cassini(shuffle=False)
    X_shuffled, y_shuffled = load_cassini(shuffle=True)

    ordered_batch_X, ordered_batch_y = first_batch(
        X_ordered, y_ordered, CASSINI_BATCH_INDEX
    )
    shuffled_batch_X, shuffled_batch_y = first_batch(
        X_shuffled, y_shuffled, CASSINI_BATCH_INDEX
    )

    x_vals = X_ordered[:, 0]
    y_vals = X_ordered[:, 1]
    x_pad = (x_vals.max() - x_vals.min()) * 0.05
    y_pad = (y_vals.max() - y_vals.min()) * 0.05
    xlim = (x_vals.min() - x_pad, x_vals.max() + x_pad)
    ylim = (y_vals.min() - y_pad, y_vals.max() + y_pad)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    plot_scatter(
        axes[0],
        X_ordered,
        y_ordered,
        "Complete Cassini",
        xlim,
        ylim,
        point_size=4,
        alpha=0.7,
    )
    plot_scatter(
        axes[1],
        ordered_batch_X,
        ordered_batch_y,
        f"Ordered batch {CASSINI_BATCH_INDEX}",
        xlim,
        ylim,
        point_size=20,
        alpha=0.9,
    )
    plot_scatter(
        axes[2],
        shuffled_batch_X,
        shuffled_batch_y,
        f"Shuffled batch {CASSINI_BATCH_INDEX}",
        xlim,
        ylim,
        point_size=20,
        alpha=0.9,
    )

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "cassini_batches.png"), bbox_inches="tight")
    plt.close(fig)


def plot_concept_drift_comparison():
    fig, axes = plt.subplots(3, 2, figsize=(14, 18), sharex=False, sharey=False)

    for row_idx, (display_name, generator_name, n_samples) in enumerate(
        CONCEPT_DRIFT_SPECS
    ):
        X_full, y_full = load_r_stream(generator_name)
        X_full = X_full[:n_samples]
        y_full = y_full[:n_samples]
        mask = y_full != -1
        X_removed = X_full[mask]
        y_removed = y_full[mask]

        x_vals = X_full[:, 0]
        y_vals = X_full[:, 1]
        x_pad = (x_vals.max() - x_vals.min()) * 0.05
        y_pad = (y_vals.max() - y_vals.min()) * 0.05
        xlim = (x_vals.min() - x_pad, x_vals.max() + x_pad)
        ylim = (y_vals.min() - y_pad, y_vals.max() + y_pad)

        plot_scatter(
            axes[row_idx, 0],
            X_removed,
            y_removed,
            f"{display_name}: noise removed",
            xlim,
            ylim,
            point_size=5,
            alpha=0.8,
        )
        plot_scatter(
            axes[row_idx, 1],
            X_full,
            y_full,
            f"{display_name}: noise relabeled to -1",
            xlim,
            ylim,
            point_size=5,
            alpha=0.8,
        )

    fig.tight_layout()
    fig.savefig(
        os.path.join(OUTPUT_DIR, "concept_drift_noise_handling.png"),
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_scalability_comparison():
    X_clean, y_clean = generate_scl(SCALABILITY_DIM, with_noise=False)
    X_noisy, y_noisy = generate_scl(SCALABILITY_DIM, with_noise=True)

    x_vals = X_noisy[:, 0]
    y_vals = X_noisy[:, 1]
    x_pad = (x_vals.max() - x_vals.min()) * 0.05
    y_pad = (y_vals.max() - y_vals.min()) * 0.05
    xlim = (x_vals.min() - x_pad, x_vals.max() + x_pad)
    ylim = (y_vals.min() - y_pad, y_vals.max() + y_pad)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
    plot_scatter(
        axes[0],
        X_clean,
        y_clean,
        f"SCL-{SCALABILITY_DIM}: without noise",
        xlim,
        ylim,
        point_size=7,
        alpha=0.8,
    )
    plot_scatter(
        axes[1],
        X_noisy,
        y_noisy,
        f"SCL-{SCALABILITY_DIM}: with noise",
        xlim,
        ylim,
        point_size=7,
        alpha=0.8,
    )

    fig.tight_layout()
    fig.savefig(
        os.path.join(OUTPUT_DIR, "scalability_noise_comparison.png"),
        bbox_inches="tight",
    )
    plt.close(fig)


def main():
    plot_cassini_comparison()
    plot_concept_drift_comparison()
    plot_scalability_comparison()
    print(f"Figures saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
