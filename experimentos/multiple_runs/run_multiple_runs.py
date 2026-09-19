import logging
import os
import sys
import time

import numpy as np
import optuna
import pandas as pd
import rpy2.rinterface_lib.callbacks
import rpy2.robjects as robjects
from MicroTEDAClus.MicroTEDAClus import MicroTEDAClus, MicroTEDAClusParameterSearchSpace
from pympler import asizeof
from rpy2.robjects.packages import importr
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score
from tqdm import tqdm

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
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

DATASET_SIZES = [4000]
FAMILIES = ["SCL2", "benchmark_1", "benchmark_2"]
DATASET_CONFIGS = [(family, n) for family in FAMILIES for n in DATASET_SIZES]

# Disjoint seed offsets so the three families never collide even though
# n_total and run_idx repeat across them.
FAMILY_SEED_OFFSET = {"SCL2": 0, "benchmark_1": 100_000_000, "benchmark_2": 200_000_000}

N_RUNS_TOTAL = 1000
SCL_DIM = 2
NOISE_RATIO = 400 / 4400  # ~0.0909, matches 10:1 ratio

SPLIT_FRAC = 0.2
BATCH_SIZE = 100
OPT_TRIALS = 100
OPT_TIMEOUT = 300  # seconds

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
    elif family == "benchmark_1":
        return get_benchmark_total(1, n_total, seed)
    elif family == "benchmark_2":
        return get_benchmark_total(2, n_total, seed)
    else:
        raise ValueError(f"Unknown family: {family}")


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

        if last_stored_run(path) >= N_RUNS_TOTAL:
            print(f"[{family}_{n_total}] {N_RUNS_TOTAL} runs reached.")


if __name__ == "__main__":
    main()
