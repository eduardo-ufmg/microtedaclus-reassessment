"""
Preprocessing-impact experiment.

Investigates how a cumulative chain of offline preprocessing steps affects
MicroTEDAClus on the `cluto-t7-10k` (chameleon ds7) dataset.

Pipeline (cumulative — each stage is derived from the previous one):

    raw -> noise_removed -> downsampled -> ordered -> scaled

Stage definitions (see chat discussion for the rationale):
    raw            : dataset as loaded from the arff file, no operation.
    noise_removed  : drop points whose ground-truth label is -1 (noise class).
    downsampled    : random uniform subsample, 50% of the previous stage's
                      points, original relative order preserved, seed=42.
    ordered        : stable sort by ground-truth label (ascending) — points
                      of the same cluster become contiguous in the stream.
    scaled         : MinMax scaling to [0, 1] per feature (column).

Optimization/evaluation protocol is kept identical to `experiment.py`:
    - 20% / 80% opt/eval split
    - Optuna study, 100 trials, 600s timeout, objective = ARI - 1e-3 * seconds
    - Batch size 100, streaming predict-then-fit loop

Difference from `experiment.py`:
    - Single run per stage (no repetition, no CSV persistence).
    - Only batchwise ARI and processing time (predict, fit) are tracked
      during evaluation (no memory / microcluster-count tracking).
    - Only the MicroTEDAClus model is evaluated.
    - Results are plotted directly, not written to disk as CSV.
"""

import os
import time
from io import StringIO

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import urllib.request
from scipy.io import arff
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from MicroTEDAClus.MicroTEDAClus import (
    MicroTEDAClus,
    MicroTEDAClusParameterSearchSpace,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)

OUTPUT_DIR = "./plots/preprocessing_study"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CHAMELEON_DS7_URL = (
    "https://raw.githubusercontent.com/deric/clustering-benchmark/master/"
    "src/main/resources/datasets/artificial/cluto-t7-10k.arff"
)

BATCH_SIZE = 100
OPT_TRIALS = 100
OPT_TIMEOUT = 600  # seconds
SPLIT_FRAC = 0.2
DOWNSAMPLE_FRAC = 0.5
RNG_SEED = 0

STAGES = ["raw", "noise_removed", "downsampled", "ordered", "scaled"]


# --------------------------------------------------------------------------
# Data loading
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


# --------------------------------------------------------------------------
# Preprocessing operators (each takes the *previous* stage's (X, y))
# --------------------------------------------------------------------------
def op_noise_removed(X, y):
    mask = y != -1
    return X[mask], y[mask]


def op_downsampled(X, y, frac=DOWNSAMPLE_FRAC, rng=None):
    rng = rng or np.random.default_rng(RNG_SEED)
    n_keep = max(1, int(round(len(X) * frac)))
    idx = rng.choice(len(X), size=n_keep, replace=False)
    idx = np.sort(idx)  # preserve relative order; "ordered" stage handles order
    return X[idx], y[idx]


def op_ordered(X, y):
    idx = np.argsort(y, kind="stable")
    return X[idx], y[idx]


def op_scaled(X, y):
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y


def build_pipeline(X0, y0):
    """Returns dict stage_name -> (X, y), built cumulatively."""
    datasets = {}
    X, y = X0, y0
    datasets["raw"] = (X, y)

    X, y = op_noise_removed(X, y)
    datasets["noise_removed"] = (X, y)

    X, y = op_downsampled(X, y)
    datasets["downsampled"] = (X, y)

    X, y = op_ordered(X, y)
    datasets["ordered"] = (X, y)

    X, y = op_scaled(X, y)
    datasets["scaled"] = (X, y)

    return datasets


# --------------------------------------------------------------------------
# Optuna objective
# --------------------------------------------------------------------------
def objective(trial, X_opt, y_opt, batch_size):
    params = MicroTEDAClusParameterSearchSpace().sample_parameters(trial)

    try:
        model = MicroTEDAClus(**params)
    except Exception as e:
        print(f"Error initializing MicroTEDAClus with params {params}: {e}")
        raise optuna.exceptions.TrialPruned()

    burn_in_idx = int(0.05 * len(X_opt))
    y_pred_all = []
    num_batches = len(X_opt) // batch_size

    t0 = time.process_time_ns()

    try:
        for b in range(num_batches):
            start_idx = b * batch_size
            end_idx = start_idx + batch_size
            X_batch = X_opt[start_idx:end_idx]

            preds = model.predict(X_batch)
            y_pred_all.extend(preds)
            model.fit(X_batch)

    except Exception as e:
        print(f"Error occurred while executing MicroTEDAClus: {e}")
        raise optuna.exceptions.TrialPruned()

    t_total_sec = (time.process_time_ns() - t0) / 1e9

    y_pred_eval = np.array(y_pred_all)[burn_in_idx:]
    y_opt_eval = y_opt[: len(y_pred_all)][burn_in_idx:]

    ari = adjusted_rand_score(y_opt_eval, y_pred_eval)

    l2_time_penalty = 1e-3 * t_total_sec
    return ari - l2_time_penalty


# --------------------------------------------------------------------------
# Evaluation loop (single run; tracks only ari, predict/fit time)
# --------------------------------------------------------------------------
def evaluate(model, X_eval, y_eval, batch_size):
    ari_list = []
    t_predict_ms = []
    t_fit_ms = []

    num_batches = len(X_eval) // batch_size

    for b in range(num_batches):
        start_idx = b * batch_size
        end_idx = start_idx + batch_size
        X_batch = X_eval[start_idx:end_idx]
        y_batch = y_eval[start_idx:end_idx]

        try:
            t0_pred = time.process_time_ns()
            preds = model.predict(X_batch)
            t_pred_ns = time.process_time_ns() - t0_pred

            t0_fit = time.process_time_ns()
            model.fit(X_batch)
            t_fit_ns = time.process_time_ns() - t0_fit

            ari = adjusted_rand_score(np.asarray(y_batch), preds)

        except Exception as e:
            print(f"Error at batch {b}: {e}")
            ari, t_pred_ns, t_fit_ns = 0.0, 0, 0

        ari_list.append(ari)
        t_predict_ms.append(t_pred_ns / 1e6)
        t_fit_ms.append(t_fit_ns / 1e6)

    return np.array(ari_list), np.array(t_predict_ms), np.array(t_fit_ms)


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------
def plot_dataset(X, y, stage, out_dir):
    fig, ax = plt.subplots()
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y)
    ax.set_title(f"{stage} (n={len(X)})")
    ax.set_xlabel("$x_0$")
    ax.set_ylabel("$x_1$")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{stage}_dataset.png"))
    plt.close(fig)


def plot_metrics(ari, t_predict_ms, t_fit_ms, stage, out_dir):
    batches = np.arange(len(ari))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(batches, ari, color="tab:blue", linewidth=2.0)
    axes[0].set_title(f"{stage} — ARI per batch")
    axes[0].set_xlabel("Batch Index")
    axes[0].set_ylabel("Adjusted Rand Index")
    axes[0].set_ylim(-1.0, 1.0)
    axes[0].grid(True, linestyle="--", alpha=0.5)

    axes[1].plot(
        batches, t_predict_ms, label="predict", color="tab:orange", linewidth=2.0
    )
    axes[1].plot(batches, t_fit_ms, label="fit", color="tab:green", linewidth=2.0)
    axes[1].set_title(f"{stage} — processing time per batch")
    axes[1].set_xlabel("Batch Index")
    axes[1].set_ylabel("Time (ms)")
    axes[1].set_yscale("log")
    axes[1].legend()
    axes[1].grid(True, which="both", linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{stage}_metrics.png"))
    plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    print("Loading chameleon ds7 (cluto-t7-10k)...")
    X0, y0 = get_chameleon(CHAMELEON_DS7_URL)

    print("Building cumulative preprocessing pipeline...")
    datasets = build_pipeline(X0, y0)

    for stage in tqdm(STAGES, desc="Stages"):
        X, y = datasets[stage]

        plot_dataset(X, y, stage, OUTPUT_DIR)

        n_samples = len(X)
        split_idx = int(SPLIT_FRAC * n_samples)
        X_opt, y_opt = X[:split_idx], y[:split_idx]
        X_eval, y_eval = X[split_idx:], y[split_idx:]

        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda trial: objective(trial, X_opt, y_opt, BATCH_SIZE),
            n_trials=OPT_TRIALS,
            timeout=OPT_TIMEOUT,
        )

        try:
            best_params = study.best_params
            model = MicroTEDAClus(**best_params)
        except ValueError:
            print(f"[{stage}] All trials pruned — skipping evaluation.")
            continue

        ari, t_predict_ms, t_fit_ms = evaluate(model, X_eval, y_eval, BATCH_SIZE)
        plot_metrics(ari, t_predict_ms, t_fit_ms, stage, OUTPUT_DIR)

    print(f"Done. Plots written to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
