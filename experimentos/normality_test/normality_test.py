"""
Normality-of-metrics experiment.

For MicroTEDAClus on six dataset configurations:

    - SCL2       (synthetic 2D Gaussian blobs + uniform noise), n in {4000, 8000}
    - Benchmark1 (R streamMOA DSD_Benchmark(1)),                n in {4000, 8000}
    - Benchmark2 (R streamMOA DSD_Benchmark(2)),                n in {4000, 8000}

this script:

  1. Runs up to 100 independent tune+evaluate repetitions per configuration.
  2. In each repetition, records four metrics measured at the evaluation
     batch immediately after the middle one: ARI, total time (predict+fit,
     ms), memory footprint (bytes), microcluster count.
  3. Persists results incrementally to CSV (results/<family>_<n>.csv), so
     the script can be interrupted and resumed: on restart it reads the
     last stored run index and continues from there, up to 100 runs.
  4. Once 100 runs exist for a given configuration, produces a 2x4 figure
     (rows = {20 runs, 100 runs}, cols = {ARI, time, memory, MC count}):
     histogram of each metric with the best-fit normal curve overlaid,
     annotated with a Shapiro-Wilk normality test result.

If a configuration already has 100 stored runs and its figure already
exists, the script does nothing for that configuration (idempotent re-run).

DATASET FAMILIES
  - SCL2: as before — make_blobs() plus a fixed-ratio (10:1 blobs:noise,
    matching the original generate_scl) block of uniform noise points,
    labeled -1.
  - Benchmark1 / Benchmark2: DSD_Benchmark(1) / DSD_Benchmark(2) from R's
    streamMOA, pulled via rpy2, `n_total` points via get_points(..., info =
    TRUE). NO synthetic noise points are added (per your instruction). The
    only correction applied is the same one used in experiment.py's
    get_r_stream: any NaN class label from R is mapped to -1 (the noise
    label), via np.nan_to_num(y, nan=-1).

ASSUMPTIONS (carried over / new, please flag if any are wrong):
  - Optuna settings reused unchanged: TPE sampler, 100 trials, 300s
    timeout, objective = ARI - 1e-3*seconds. With up to 100 runs x 6
    configurations, this script can take a very long time; it is designed
    to be resumable, not to finish in one sitting.
  - "One after the middle batch" is taken over the EVALUATION stream
    (X_eval, using the tuned best model), matching the structure of
    experiment.py/preproc_experiment.py's evaluation loop.
    target_batch = num_eval_batches // 2 + 1 (clamped if too small).
  - 20% / 80% opt/eval split, batch size 100 (as in the reference scripts).
  - Runs are seeded deterministically per (family, n_total, run_index):
    for SCL2 this seeds a numpy Generator; for the Benchmark families it
    seeds R via set.seed() before DSD_Benchmark/get_points, so reruns are
    reproducible across resumed sessions.
  - A run that raises an exception during tuning/evaluation is still
    recorded (status="ERROR: ...", metrics NaN) and counts toward the run
    index, so a persistently failing configuration doesn't stall forever.

If any of these assumptions are wrong, edit the CONFIG block below before
running.
"""

import logging
import os
import sys
import time

import numpy as np
import optuna
import pandas as pd
import rpy2.rinterface_lib.callbacks
import rpy2.robjects as robjects
from pympler import asizeof
from rpy2.robjects.packages import importr
from scipy import stats
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score
from tqdm import tqdm
import matplotlib.pyplot as plt

from MicroTEDAClus.MicroTEDAClus import (
    MicroTEDAClus,
    MicroTEDAClusParameterSearchSpace,
)

rpy2.rinterface_lib.callbacks.logger.setLevel(logging.ERROR)
optuna.logging.set_verbosity(optuna.logging.WARNING)

try:
    stream = importr("stream")
    streamMOA = importr("streamMOA")
except Exception as e:
    print(f"Critical Error loading R dependencies via rpy2: {e}")
    sys.exit(1)

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------
RESULTS_DIR = "./results"

DATASET_SIZES = [4000, 8000]
FAMILIES = ["SCL2", "Benchmark1", "Benchmark2"]
DATASET_CONFIGS = [(family, n) for family in FAMILIES for n in DATASET_SIZES]

# Disjoint seed offsets so the three families never collide even though
# n_total and run_idx repeat across them.
FAMILY_SEED_OFFSET = {"SCL2": 0, "Benchmark1": 100_000_000, "Benchmark2": 200_000_000}

ROW_RUN_COUNTS = [20, 100]  # rows of the summary figure
N_RUNS_TOTAL = 100
SCL_DIM = 2
NOISE_RATIO = 400 / 4400  # ~0.0909, matches generate_scl's 10:1 ratio

SPLIT_FRAC = 0.2
BATCH_SIZE = 100
OPT_TRIALS = 100
OPT_TIMEOUT = 300  # seconds

METRICS = ["ari", "total_time_ms", "memory_bytes", "microcluster_count"]
METRIC_LABELS = {
    "ari": "ARI",
    "total_time_ms": "Total time (ms)",
    "memory_bytes": "Memory (MB)",
    "microcluster_count": "Microcluster count",
}

os.makedirs(RESULTS_DIR, exist_ok=True)


# --------------------------------------------------------------------------
# Data generation
# --------------------------------------------------------------------------
def generate_scl_total(n_total, dim=SCL_DIM, noise_ratio=NOISE_RATIO, rng=None):
    """SCL-<dim> dataset with `n_total` points total (blobs + uniform noise),
    noise_ratio held fixed and proportional to n_total."""
    rng = rng if rng is not None else np.random.default_rng()

    blob_n = int(round(n_total * (1 - noise_ratio)))
    noise_n = n_total - blob_n

    X, y = make_blobs(
        n_samples=blob_n,
        centers=3,
        n_features=dim,
        cluster_std=0.5,
        random_state=int(rng.integers(0, 2**31 - 1)),
    )[0:2]

    noise = rng.uniform(
        low=np.min(X, axis=0), high=np.max(X, axis=0), size=(noise_n, dim)
    )
    X_full = np.vstack([X, noise])
    y_full = np.concatenate([y, np.full(noise_n, -1)])

    idx = rng.permutation(len(X_full))
    return X_full[idx], y_full[idx]


def get_benchmark_total(benchmark_id, n_total, seed):
    """DSD_Benchmark(<benchmark_id>) from R streamMOA, n_total points,
    no synthetic noise added. NaN class labels (R's noise marker) are
    mapped to -1, matching experiment.py's get_r_stream."""
    robjects.r(f"set.seed({seed})")
    robjects.r(f"dsd <- DSD_Benchmark({benchmark_id})")
    robjects.r(f"data <- get_points(dsd, n = {n_total}, info = TRUE)")
    X = np.array(robjects.r("as.matrix(data[, -ncol(data)])"))
    y = np.array(robjects.r("as.numeric(data[, ncol(data)])"))
    y = np.nan_to_num(y, nan=-1).astype(int)
    return X, y


def make_dataset(family, n_total, seed):
    if family == "SCL2":
        rng = np.random.default_rng(seed)
        return generate_scl_total(n_total, rng=rng)
    elif family == "Benchmark1":
        return get_benchmark_total(1, n_total, seed)
    elif family == "Benchmark2":
        return get_benchmark_total(2, n_total, seed)
    else:
        raise ValueError(f"Unknown family: {family}")


# --------------------------------------------------------------------------
# Optuna objective (identical structure to the reference scripts)
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
# Single tune + evaluate repetition
# --------------------------------------------------------------------------
def run_once(family, n_total, run_idx):
    seed = FAMILY_SEED_OFFSET[family] + 1000 * n_total + run_idx

    X, y = make_dataset(family, n_total, seed)
    split_idx = int(SPLIT_FRAC * len(X))
    X_opt, y_opt = X[:split_idx], y[:split_idx]
    X_eval, y_eval = X[split_idx:], y[split_idx:]

    num_eval_batches = len(X_eval) // BATCH_SIZE
    if num_eval_batches < 1:
        return {
            "ari": np.nan,
            "total_time_ms": np.nan,
            "memory_bytes": np.nan,
            "microcluster_count": np.nan,
            "status": "ERROR: not enough eval batches",
        }
    target_batch = min(num_eval_batches // 2 + 1, num_eval_batches - 1)

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    try:
        study.optimize(
            lambda trial: objective(trial, X_opt, y_opt, BATCH_SIZE),
            n_trials=OPT_TRIALS,
            timeout=OPT_TIMEOUT,
        )
        best_params = study.best_params
        model = MicroTEDAClus(**best_params)
    except ValueError:
        return {
            "ari": np.nan,
            "total_time_ms": np.nan,
            "memory_bytes": np.nan,
            "microcluster_count": np.nan,
            "status": "ERROR: all trials pruned",
        }
    except Exception as e:
        return {
            "ari": np.nan,
            "total_time_ms": np.nan,
            "memory_bytes": np.nan,
            "microcluster_count": np.nan,
            "status": f"ERROR: {str(e).replace(',', ';')}",
        }

    try:
        for b in range(target_batch + 1):
            start_idx = b * BATCH_SIZE
            end_idx = start_idx + BATCH_SIZE
            X_batch = X_eval[start_idx:end_idx]
            y_batch = y_eval[start_idx:end_idx]

            t0_pred = time.process_time_ns()
            preds = model.predict(X_batch)
            t_pred_ns = time.process_time_ns() - t0_pred

            t0_fit = time.process_time_ns()
            model.fit(X_batch)
            t_fit_ns = time.process_time_ns() - t0_fit

            if b == target_batch:
                ari = adjusted_rand_score(np.asarray(y_batch), preds)
                total_time_ms = (t_pred_ns + t_fit_ns) / 1e6
                memory_bytes = asizeof.asizeof(model)
                mc_count = model.microcluster_count()

        return {
            "ari": ari,
            "total_time_ms": total_time_ms,
            "memory_bytes": memory_bytes,
            "microcluster_count": mc_count,
            "status": "OK",
        }
    except Exception as e:
        return {
            "ari": np.nan,
            "total_time_ms": np.nan,
            "memory_bytes": np.nan,
            "microcluster_count": np.nan,
            "status": f"ERROR: {str(e).replace(',', ';')}",
        }


# --------------------------------------------------------------------------
# Persistency helpers
# --------------------------------------------------------------------------
def csv_path(family, n_total):
    return os.path.join(RESULTS_DIR, f"{family}_{n_total}.csv")


def fig_path(family, n_total):
    return os.path.join(RESULTS_DIR, f"{family}_{n_total}_normality.png")


def last_stored_run(path):
    if not os.path.exists(path):
        return 0
    existing = pd.read_csv(path, index_col=0)
    if existing.empty:
        return 0
    return int(existing.index.max())


def append_row(path, run_idx, row_dict):
    header = not os.path.exists(path)
    row_df = pd.DataFrame([row_dict], index=[run_idx])
    row_df.index.name = "run"
    row_df.to_csv(path, mode="a", header=header)


# --------------------------------------------------------------------------
# Figure generation
# --------------------------------------------------------------------------
def make_normality_figure(family, n_total):
    df = pd.read_csv(csv_path(family, n_total), index_col=0)

    fig, axes = plt.subplots(len(ROW_RUN_COUNTS), len(METRICS), figsize=(20, 9))

    for row, n_runs in enumerate(ROW_RUN_COUNTS):
        subset = df.loc[df.index <= n_runs]
        for col, metric in enumerate(METRICS):
            ax = axes[row, col]
            data = subset[metric].dropna().to_numpy()

            if len(data) < 3:
                ax.set_title(f"n={n_runs}: insufficient valid data")
                continue

            if metric == "memory_bytes":
                data = data / (1024 * 1024)  # Convert bytes to MB

            ax.hist(data, bins="auto", density=True, alpha=0.6, color="tab:blue")

            mu, sigma = stats.norm.fit(data)
            x = np.linspace(data.min(), data.max(), 200)
            ax.plot(
                x,
                stats.norm.pdf(x, mu, sigma),
                "r-",
                linewidth=2,
                label="fitted normal",
            )

            w_stat, p_value = stats.shapiro(data)

            ax.set_title(
                f"{METRIC_LABELS[metric]}\n"
                f"n={n_runs}, mu={mu:.3g}, sigma={sigma:.3g}\n"
                f"Shapiro W={w_stat:.3g}, p={p_value:.3g}"
            )
            ax.set_xlabel(METRIC_LABELS[metric])
            ax.set_ylabel("Density")
            ax.legend(fontsize=7)
            ax.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle(f"{family}, {n_total} samples — middle batch")
    plt.tight_layout()
    fig.savefig(fig_path(family, n_total))
    plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    for family, n_total in DATASET_CONFIGS:
        path = csv_path(family, n_total)
        last_run = last_stored_run(path)
        resume_from = last_run + 1

        if resume_from > N_RUNS_TOTAL:
            print(f"[{family}_{n_total}] Already has {last_run} runs — not running.")
        else:
            for run_idx in tqdm(
                range(resume_from, N_RUNS_TOTAL + 1),
                desc=f"{family}_{n_total} runs",
                initial=resume_from - 1,
                total=N_RUNS_TOTAL,
            ):
                row = run_once(family, n_total, run_idx)
                append_row(path, run_idx, row)

        # Generate the figure once 100 runs exist and it hasn't been made yet.
        if last_stored_run(path) >= N_RUNS_TOTAL and not os.path.exists(
            fig_path(family, n_total)
        ):
            print(
                f"[{family}_{n_total}] 100 runs reached — generating normality figure."
            )
            make_normality_figure(family, n_total)


if __name__ == "__main__":
    main()
