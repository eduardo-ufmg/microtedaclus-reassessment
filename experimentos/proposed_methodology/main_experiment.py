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

from models.AutoCloud.AutoCloud import AutoCloud, AutoCloudParameterSearchSpace
from models.CEDAS.CEDAS import CEDAS, CEDASParameterSearchSpace
from models.CluStream.CluStream import CluStream, CluStreamParameterSearchSpace
from models.DenStream.DenStream import DenStream, DenStreamParameterSearchSpace
from models.MicroTEDAClus.MicroTEDAClus import (
    MicroTEDAClus,
    MicroTEDAClusParameterSearchSpace,
)
from models.MicroTEDAClusCorrected.MicroTEDAClusCorrected import (
    MicroTEDAClusCorrected,
    MicroTEDAClusCorrectedParameterSearchSpace,
)
from models.StreamKM.StreamKM import StreamKM, StreamKMParameterSearchSpace
from models.TEDACloud.TEDACloud import TEDACloud, TEDACloudParameterSearchSpace
from editorial_config import DATASET_ORDER

rpy2.rinterface_lib.callbacks.logger.setLevel(logging.ERROR)
optuna.logging.set_verbosity(optuna.logging.WARNING)

DIR_RESULTS = "./results"
DIR_PARAMS = "./results_params"
DIR_SAMPLES = "./data_samples"
os.makedirs(DIR_RESULTS, exist_ok=True)
os.makedirs(DIR_PARAMS, exist_ok=True)
os.makedirs(DIR_SAMPLES, exist_ok=True)

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
RUNS = 94  # Aligned with k* = 94
BATCH_SIZE = 100
OPT_TRIALS = 100
OPT_TIMEOUT = 300
BURN_IN_FRACTION = 0.05
SPLIT_FRACTION = 0.2
BASE_SEED = 0

DATASETS = [
    *DATASET_ORDER,
    "SCL-1",
    "SCL-2",
    "SCL-4",
    "SCL-8",
    "SCL-16",
    "SCL-32",
    "SCL-64",
]

MODEL_MAP = {
    "CluStream": (CluStream, CluStreamParameterSearchSpace),
    "DenStream": (DenStream, DenStreamParameterSearchSpace),
    "MicroTEDAClus": (MicroTEDAClus, MicroTEDAClusParameterSearchSpace),
    "MicroTEDAClusCorrected": (
        MicroTEDAClusCorrected,
        MicroTEDAClusCorrectedParameterSearchSpace,
    ),
    "CEDAS": (CEDAS, CEDASParameterSearchSpace),
    "TEDACloud": (TEDACloud, TEDACloudParameterSearchSpace),
    "AutoCloud": (AutoCloud, AutoCloudParameterSearchSpace),
    "StreamKM": (StreamKM, StreamKMParameterSearchSpace),
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

    if name == "cassini":
        robjects.r("data <- mlbench.cassini(4000)")
        X = np.array(robjects.r("data$x"))
        y = np.nan_to_num(
            np.array(robjects.r("as.numeric(data$classes)")), nan=-1
        ).astype(int)

        rng = np.random.default_rng(seed)
        shuffle_idx = rng.permutation(len(X))
        return X[shuffle_idx], y[shuffle_idx]

    elif name == "random_rbf_events":
        robjects.r("""
            dsd <- DSD_RandomRBFGeneratorEvents(eventFrequency=1000L, eventMergeSplitOption=TRUE, eventDeleteCreate=TRUE)
            data <- get_points(dsd, n=10000, info=TRUE)
        """)
    elif name == "benchmark_1":
        robjects.r(
            "dsd <- DSD_Benchmark(1)\ndata <- get_points(dsd, n=4000, info=TRUE)"
        )
    elif name == "benchmark_2":
        robjects.r(
            "dsd <- DSD_Benchmark(2)\ndata <- get_points(dsd, n=4000, info=TRUE)"
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
    if dataset_name == "chameleon_ds4":
        return get_chameleon(
            "https://raw.githubusercontent.com/deric/clustering-benchmark/master/src/main/resources/datasets/artificial/cluto-t8-8k.arff"
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
    # Flatten loop for atomic progress tracking
    tasks = list(itertools.product(DATASETS, MODEL_MAP.keys(), range(RUNS)))

    with tqdm(total=len(tasks), desc="Experiments") as pbar:
        for dataset, model_name, run in tasks:
            output_file = os.path.join(DIR_RESULTS, f"{dataset}_{model_name}_{run}.csv")
            params_file = os.path.join(DIR_PARAMS, f"{dataset}_{model_name}_{run}.csv")

            # Persistency constraint check
            if os.path.exists(output_file) and os.path.exists(params_file):
                pbar.update(1)
                continue

            dataset_hash = zlib.crc32(dataset.encode("utf-8"))
            run_seed = BASE_SEED + run + (dataset_hash % 10000)
            X, y = get_dataset(dataset, run_seed)
            split_idx = int(SPLIT_FRACTION * len(X))
            X_opt, y_opt = X[:split_idx], y[:split_idx]
            X_eval, y_eval = X[split_idx:], y[split_idx:]

            # Isolate Data Sampling (10% per batch constraint) per dataset per run
            sample_file_path = os.path.join(DIR_SAMPLES, f"{dataset}_{run}.csv")
            if not os.path.exists(sample_file_path):
                rng = np.random.default_rng(run_seed)
                sample_indices = []
                for b in range(len(X_eval) // BATCH_SIZE):
                    start_idx = b * BATCH_SIZE
                    sample_size = max(1, BATCH_SIZE // 10)
                    idx = rng.choice(BATCH_SIZE, size=sample_size, replace=False)
                    sample_indices.extend(start_idx + idx)

                sample_indices = np.sort(sample_indices)
                global_indices = sample_indices + split_idx
                sampled_data = np.column_stack(
                    (global_indices, X_eval[sample_indices], y_eval[sample_indices])
                )
                cols = (
                    ["sample_idx"] + [f"x{i}" for i in range(X_eval.shape[1])] + ["y"]
                )
                pd.DataFrame(sampled_data, columns=cols).to_csv(
                    sample_file_path, index=False
                )

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
                    break  # Stop evaluation for this run constraint

            pd.DataFrame(results).to_csv(output_file, index=False)
            pbar.update(1)


if __name__ == "__main__":
    main()
