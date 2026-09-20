# iterative_validation_experiment.py

import itertools
import logging
import os
import sys
import time
import urllib.request
import zlib
from io import StringIO

import numpy as np
import optuna
import pandas as pd
import rpy2.rinterface_lib.callbacks
import rpy2.robjects as robjects
from pympler import asizeof
from rpy2.robjects.packages import importr
from scipy.io import arff
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score
from tqdm import tqdm

from models.MicroTEDAClus.MicroTEDAClus import (
    MicroTEDAClus,
    MicroTEDAClusParameterSearchSpace,
)
from models.MicroTEDAClusCorrected.MicroTEDAClusCorrected import (
    MicroTEDAClusCorrected,
    MicroTEDAClusCorrectedParameterSearchSpace,
)

rpy2.rinterface_lib.callbacks.logger.setLevel(logging.ERROR)
optuna.logging.set_verbosity(optuna.logging.WARNING)

DIR_RESULTS = "./results_iterative"
DIR_PARAMS = "./results_params_iterative"
os.makedirs(DIR_RESULTS, exist_ok=True)
os.makedirs(DIR_PARAMS, exist_ok=True)

try:
    stream = importr("stream")
    streamMOA = importr("streamMOA")
    mlbench = importr("mlbench")
except Exception as e:
    print(f"Critical Error loading R dependencies via rpy2: {e}")
    sys.exit(1)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
RUNS = 32
BATCH_SIZE = 100
OPT_TRIALS = 64
OPT_TIMEOUT = 180
BURN_IN_FRACTION = 0.05
SPLIT_FRACTION = 0.2
BASE_SEED = 0

DATASETS = [
    "benchmark_1",
    "chameleon_ds3",
    "SCL-8",
]

MODEL_MAP = {
    "MicroTEDAClus": (MicroTEDAClus, MicroTEDAClusParameterSearchSpace),
    "MicroTEDAClusCorrected": (
        MicroTEDAClusCorrected,
        MicroTEDAClusCorrectedParameterSearchSpace,
    ),
}


# -----------------------------------------------------------------------------
# Deterministic Data Generation
# -----------------------------------------------------------------------------
def get_chameleon(url: str) -> tuple[np.ndarray, np.ndarray]:
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as response:
        data, _ = arff.loadarff(StringIO(response.read().decode("utf-8")))

    df = pd.DataFrame(data)
    target_col = df.columns[-1]
    y_raw = df[target_col].str.decode("utf-8").replace("noise", "-1")
    X = df.drop(columns=[target_col]).to_numpy()
    y = y_raw.astype(int).to_numpy()
    return X, y


def get_r_stream(name: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    robjects.r(f"set.seed({seed})")

    if name == "benchmark_1":
        robjects.r(
            "dsd <- DSD_Benchmark(1)\ndata <- get_points(dsd, n=4000, info=TRUE)"
        )
    else:
        raise ValueError(f"Unknown R dataset: {name}")

    X = np.array(robjects.r("as.matrix(data[, -ncol(data)])"))
    y = np.nan_to_num(
        np.array(robjects.r("as.numeric(data[, ncol(data)])")), nan=-1
    ).astype(int)
    return X, y


def generate_scl(dim: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X, y = make_blobs(
        n_samples=4000, centers=3, n_features=dim, cluster_std=0.5, random_state=seed
    )[0:2]
    noise = rng.uniform(low=np.min(X, axis=0), high=np.max(X, axis=0), size=(400, dim))

    X_full = np.vstack([X, noise])
    y_full = np.concatenate([y, np.full(400, -1)])

    idx = rng.permutation(len(X_full))
    return X_full[idx], y_full[idx]


def get_dataset(dataset_name: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if dataset_name.startswith("SCL-"):
        return generate_scl(int(dataset_name.split("-")[1]), seed)
    if dataset_name == "chameleon_ds3":
        return get_chameleon(
            "https://raw.githubusercontent.com/deric/clustering-benchmark/master/src/main/resources/datasets/artificial/cluto-t7-10k.arff"
        )
    return get_r_stream(dataset_name, seed)


# -----------------------------------------------------------------------------
# Evaluation Strategy
# -----------------------------------------------------------------------------
def objective(trial, model_class, search_space_class, X_opt, y_opt):
    params = search_space_class().sample_parameters(trial)
    try:
        model = model_class(**params)
    except Exception:
        raise optuna.exceptions.TrialPruned()

    burn_in_idx = int(BURN_IN_FRACTION * len(X_opt))
    y_pred_all = []
    num_batches = len(X_opt) // BATCH_SIZE

    t0 = time.process_time_ns()
    try:
        for b in range(num_batches):
            start_idx, end_idx = b * BATCH_SIZE, (b + 1) * BATCH_SIZE
            X_batch = X_opt[start_idx:end_idx]

            preds = model.predict(X_batch)
            y_pred_all.extend(preds)
            model.fit(X_batch)
    except Exception:
        raise optuna.exceptions.TrialPruned()

    t_total_sec = (time.process_time_ns() - t0) / 1e9

    y_pred_eval = np.array(y_pred_all)[burn_in_idx:]
    y_opt_eval = y_opt[: len(y_pred_all)][burn_in_idx:]

    ari = adjusted_rand_score(y_opt_eval, y_pred_eval)
    return ari - (1e-3 * t_total_sec)


# -----------------------------------------------------------------------------
# Main Execution Process
# -----------------------------------------------------------------------------
def main():
    tasks = list(itertools.product(DATASETS, MODEL_MAP.keys(), range(RUNS)))

    with tqdm(total=len(tasks), desc="Experiments") as pbar:
        for dataset, model_name, run in tasks:
            output_file = os.path.join(DIR_RESULTS, f"{dataset}_{model_name}_{run}.csv")
            params_file = os.path.join(DIR_PARAMS, f"{dataset}_{model_name}_{run}.csv")

            if os.path.exists(output_file) and os.path.exists(params_file):
                pbar.update(1)
                continue

            dataset_hash = zlib.crc32(dataset.encode("utf-8"))
            run_seed = BASE_SEED + run + (dataset_hash % 10000)
            X, y = get_dataset(dataset, run_seed)
            split_idx = int(SPLIT_FRACTION * len(X))
            X_opt, y_opt = X[:split_idx], y[:split_idx]
            X_eval, y_eval = X[split_idx:], y[split_idx:]

            model_class, search_space_class = MODEL_MAP[model_name]
            sampler = optuna.samplers.TPESampler(seed=run_seed)
            study = optuna.create_study(direction="maximize", sampler=sampler)
            study.optimize(
                lambda trial: objective(
                    trial, model_class, search_space_class, X_opt, y_opt
                ),
                n_trials=OPT_TRIALS,
                timeout=OPT_TIMEOUT,
            )

            try:
                best_params = study.best_params
                model = model_class(**best_params)
                params_records = [
                    {
                        "dataset": dataset,
                        "model": model_name,
                        "run": run,
                        "param_name": k,
                        "param_value": v,
                    }
                    for k, v in best_params.items()
                ]
            except ValueError:
                pd.DataFrame(
                    [{"batch": 0, "status": "ERROR: All trials pruned"}]
                ).to_csv(output_file, index=False)
                pd.DataFrame([]).to_csv(params_file, index=False)
                pbar.update(1)
                continue

            pd.DataFrame(params_records).to_csv(params_file, index=False)

            results = []
            num_batches = len(X_eval) // BATCH_SIZE

            for b in range(num_batches):
                start_idx, end_idx = b * BATCH_SIZE, (b + 1) * BATCH_SIZE
                X_batch, y_batch = X_eval[start_idx:end_idx], y_eval[start_idx:end_idx]

                try:
                    t0_pred_wall = time.perf_counter_ns()
                    t0_pred_cpu = time.process_time_ns()
                    preds = model.predict(X_batch)
                    t_pred_cpu_ns = time.process_time_ns() - t0_pred_cpu
                    t_pred_wall_ns = time.perf_counter_ns() - t0_pred_wall

                    t0_fit_wall = time.perf_counter_ns()
                    t0_fit_cpu = time.process_time_ns()
                    model.fit(X_batch)
                    t_fit_cpu_ns = time.process_time_ns() - t0_fit_cpu
                    t_fit_wall_ns = time.perf_counter_ns() - t0_fit_wall

                    ari = adjusted_rand_score(np.asarray(y_batch), preds)
                    mem_bytes = asizeof.asizeof(model)
                    mc_count = model.microcluster_count()
                    status = "OK"
                except Exception as e:
                    ari = t_pred_cpu_ns = t_fit_cpu_ns = t_pred_wall_ns = (
                        t_fit_wall_ns
                    ) = mem_bytes = mc_count = 0
                    status = f"ERROR: {str(e).replace(',', ';')}"

                results.append(
                    {
                        "batch": b,
                        "ari": ari,
                        "time_predict_cpu_ns": t_pred_cpu_ns,
                        "time_fit_cpu_ns": t_fit_cpu_ns,
                        "time_predict_wall_ns": t_pred_wall_ns,
                        "time_fit_wall_ns": t_fit_wall_ns,
                        "microcluster_count": mc_count,
                        "memory_bytes": mem_bytes,
                        "status": status,
                    }
                )

                if status.startswith("ERROR"):
                    break

            pd.DataFrame(results).to_csv(output_file, index=False)
            pbar.update(1)


if __name__ == "__main__":
    main()
