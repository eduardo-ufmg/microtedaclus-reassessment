import os
from io import StringIO
import urllib.request

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import arff

OUTPUT_DIR = "./plots/batch_leakage_study"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CHAMELEON_DS7_URL = (
    "https://raw.githubusercontent.com/deric/clustering-benchmark/master/"
    "src/main/resources/datasets/artificial/cluto-t7-10k.arff"
)

BATCH_SIZE = 100
DOWNSAMPLE_FRAC = 0.5
RNG_SEED = 0
TARGET_BATCHES = [0, 10, 20, 30]


# --------------------------------------------------------------------------
# Data Loading and Preprocessing
# --------------------------------------------------------------------------
def get_chameleon(url):
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as response:
        data, meta = arff.loadarff(StringIO(response.read().decode("utf-8")))

    df = pd.DataFrame(data)
    target_col = df.columns[-1]
    y_raw = df[target_col].str.decode("utf-8")
    y_raw = y_raw.replace("noise", "-1")
    y = y_raw.astype(int).values
    X = df.drop(columns=[target_col]).values
    return X, y


def op_noise_removed(X, y):
    mask = y != -1
    return X[mask], y[mask]


def op_downsampled(X, y, frac=DOWNSAMPLE_FRAC, rng=None):
    rng = rng or np.random.default_rng(RNG_SEED)
    n_keep = max(1, int(round(len(X) * frac)))
    idx = rng.choice(len(X), size=n_keep, replace=False)
    idx = np.sort(idx)  # Preserves chronology
    return X[idx], y[idx]


# --------------------------------------------------------------------------
# Plotting routine
# --------------------------------------------------------------------------
def plot_batch_grid(
    X, y, batch_indices, batch_size, title, filename, global_X, global_y
):
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()

    # Enforce global axis limits for contextual scale
    x0_min, x0_max = global_X[:, 0].min(), global_X[:, 0].max()
    x1_min, x1_max = global_X[:, 1].min(), global_X[:, 1].max()

    pad_x0 = (x0_max - x0_min) * 0.05
    pad_x1 = (x1_max - x1_min) * 0.05

    # Enforce global colormap scaling
    c_min, c_max = global_y.min(), global_y.max()

    for i, b_idx in enumerate(batch_indices):
        ax = axes[i]
        start_idx = b_idx * batch_size
        end_idx = start_idx + batch_size

        X_batch = X[start_idx:end_idx]
        y_batch = y[start_idx:end_idx]

        if len(X_batch) == 0:
            ax.set_title(f"Batch {b_idx} (Empty Dataset)")
            continue

        ax.scatter(
            X_batch[:, 0],
            X_batch[:, 1],
            c=y_batch,
            cmap="tab20",
            vmin=c_min,
            vmax=c_max,
        )
        ax.set_title(f"Batch {b_idx}")
        ax.set_xlim(x0_min - pad_x0, x0_max + pad_x0)
        ax.set_ylim(x1_min - pad_x1, x1_max + pad_x1)
        ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close(fig)


# --------------------------------------------------------------------------
# Execution
# --------------------------------------------------------------------------
def main():
    print("Loading chameleon ds7 (cluto-t7-10k)...")
    X0, y0 = get_chameleon(CHAMELEON_DS7_URL)

    X_nr, y_nr = op_noise_removed(X0, y0)
    X_ds, y_ds = op_downsampled(X_nr, y_nr)

    # Induce data leakage via label-based sorting
    sorted_idx = np.argsort(y_ds)
    X_ordered = X_ds[sorted_idx]
    y_ordered = y_ds[sorted_idx]

    print("Generating unordered batches plot...")
    plot_batch_grid(
        X_ds,
        y_ds,
        TARGET_BATCHES,
        BATCH_SIZE,
        "Original Chronology",
        "unordered_batches.png",
        global_X=X_ds,
        global_y=y_ds,
    )

    print("Generating ordered batches plot...")
    plot_batch_grid(
        X_ordered,
        y_ordered,
        TARGET_BATCHES,
        BATCH_SIZE,
        "Ordered Dataset",
        "ordered_batches.png",
        global_X=X_ds,
        global_y=y_ds,
    )

    print(f"Routines finished. Figures saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
