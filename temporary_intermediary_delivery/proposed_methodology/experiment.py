import logging
import os
import sys
import time
import urllib.request
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
from models.TEDACloud.TEDACloud import TEDACloud, TEDACloudParameterSearchSpace
from models.StreamKM.StreamKM import StreamKM, StreamKMParameterSearchSpace

rpy2.rinterface_lib.callbacks.logger.setLevel(logging.ERROR)

optuna.logging.set_verbosity(optuna.logging.WARNING)
os.makedirs("./results", exist_ok=True)
os.makedirs("./data_samples", exist_ok=True)

try:
    stream = importr("stream")
    streamMOA = importr("streamMOA")
    mlbench = importr("mlbench")
except Exception as e:
    print(f"Critical Error loading R dependencies via rpy2: {e}")
    sys.exit(1)


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


def get_r_stream(name):
    if name == "cassini":
        robjects.r("data <- mlbench.cassini(4000)")
        X = np.array(robjects.r("data$x"))
        y = np.array(robjects.r("as.numeric(data$classes)"))
        y = np.nan_to_num(y, nan=-1).astype(int)
        shuffle_idx = np.random.permutation(len(X))
        return X[shuffle_idx], y[shuffle_idx]
    elif name == "RandomRBFGeneratorEvents":
        robjects.r("""
            dsd <- DSD_RandomRBFGeneratorEvents(
                eventFrequency = 1000L,
                eventMergeSplitOption = TRUE,
                eventDeleteCreate = TRUE
            )
            data <- get_points(dsd, n = 10000, info = TRUE)
        """)
        X = np.array(robjects.r("as.matrix(data[, -ncol(data)])"))
        y = np.array(robjects.r("as.numeric(data[, ncol(data)])"))
        y = np.nan_to_num(y, nan=-1).astype(int)
        return X, y
    elif name == "Benchmark(1)":
        robjects.r("dsd <- DSD_Benchmark(1)")
        robjects.r("data <- get_points(dsd, n = 4000, info = TRUE)")
        X = np.array(robjects.r("as.matrix(data[, -ncol(data)])"))
        y = np.array(robjects.r("as.numeric(data[, ncol(data)])"))
        y = np.nan_to_num(y, nan=-1).astype(int)
        return X, y
    elif name == "Benchmark(2)":
        robjects.r("dsd <- DSD_Benchmark(2)")
        robjects.r("data <- get_points(dsd, n = 4000, info = TRUE)")
        X = np.array(robjects.r("as.matrix(data[, -ncol(data)])"))
        y = np.array(robjects.r("as.numeric(data[, ncol(data)])"))
        y = np.nan_to_num(y, nan=-1).astype(int)
        return X, y
    else:
        raise ValueError(f"Unknown R dataset: {name}")


def generate_scl(dim):
    X, y = make_blobs(n_samples=4000, centers=3, n_features=dim, cluster_std=0.5)[0:2]
    noise = np.random.uniform(
        low=np.min(X, axis=0), high=np.max(X, axis=0), size=(400, dim)
    )
    X_full = np.vstack([X, noise])
    y_full = np.concatenate([y, np.full(400, -1)])
    idx = np.random.permutation(len(X_full))
    return X_full[idx], y_full[idx]


def get_dataset(dataset_name):
    if dataset_name.startswith("SCL-"):
        dim = int(dataset_name.split("-")[1])
        return generate_scl(dim)
    elif dataset_name == "chameleon_ds7":
        return get_chameleon(
            "https://raw.githubusercontent.com/deric/clustering-benchmark/master/src/main/resources/datasets/artificial/cluto-t7-10k.arff"
        )
    elif dataset_name == "chameleon_ds8":
        return get_chameleon(
            "https://raw.githubusercontent.com/deric/clustering-benchmark/master/src/main/resources/datasets/artificial/cluto-t8-8k.arff"
        )
    else:
        return get_r_stream(dataset_name)


def objective(trial, model_class, search_space_class, X_opt, y_opt, batch_size):
    params = search_space_class().sample_parameters(trial)

    try:
        model = model_class(**params)
    except Exception as e:
        print(
            f"Error initializing model {model_class.__name__} with params {params}: {e}"
        )
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
        print(f"Error occurred while fitting model {model_class.__name__}: {e}")
        raise optuna.exceptions.TrialPruned()

    t_total_sec = (time.process_time_ns() - t0) / 1e9

    y_pred_eval = np.array(y_pred_all)[burn_in_idx:]
    y_opt_eval = y_opt[: len(y_pred_all)][burn_in_idx:]

    ari = adjusted_rand_score(y_opt_eval, y_pred_eval)

    # Regularization: Penalize ARI by 0.001 per second
    l2_time_penalty = 1e-3 * t_total_sec
    return ari - l2_time_penalty


DATASETS = [
    "cassini",
    "chameleon_ds7",
    "chameleon_ds8",
    "RandomRBFGeneratorEvents",
    "Benchmark(1)",
    "Benchmark(2)",
    "SCL-1",
    "SCL-2",
    "SCL-4",
    "SCL-8",
    "SCL-16",
    "SCL-32",
    "SCL-64",
    "SCL-128",
]

MODEL_MAP = {
    # "CluStream": (CluStream, CluStreamParameterSearchSpace),
    # "DenStream": (DenStream, DenStreamParameterSearchSpace),
    # "MicroTEDAClus": (MicroTEDAClus, MicroTEDAClusParameterSearchSpace),
    # "CEDAS": (CEDAS, CEDASParameterSearchSpace),
    # "TEDACloud": (TEDACloud, TEDACloudParameterSearchSpace),
    # "AutoCloud": (AutoCloud, AutoCloudParameterSearchSpace),
    "StreamKM": (StreamKM, StreamKMParameterSearchSpace),
}

RUNS = 10
BATCH_SIZE = 100

for dataset in tqdm(DATASETS, desc="Datasets"):
    for run in tqdm(range(RUNS), desc=f"Runs for {dataset}", leave=False):
        X, y = get_dataset(dataset)
        n_samples = len(X)
        split_idx = int(0.2 * n_samples)

        X_opt, y_opt = X[:split_idx], y[:split_idx]
        X_eval, y_eval = X[split_idx:], y[split_idx:]

        # 1. Decoupled Data Sampling (10% per batch)
        sample_file_path = f"./data_samples/{dataset}_{run}.csv"
        n_eval = len(X_eval)
        num_batches = n_eval // BATCH_SIZE
        sample_indices = []

        for b in range(num_batches):
            start_idx = b * BATCH_SIZE
            sample_size = max(1, BATCH_SIZE // 10)
            idx = np.random.choice(BATCH_SIZE, size=sample_size, replace=False)
            sample_indices.extend(start_idx + idx)

        sample_indices = np.sort(sample_indices)
        global_indices = sample_indices + split_idx

        # 2. Vectorized array construction and I/O
        sampled_data = np.column_stack(
            (
                global_indices,
                np.asarray(X_eval[sample_indices]),
                np.asarray(y_eval[sample_indices]),
            )
        )

        col_names = ["sample_idx"] + [f"x{i}" for i in range(X_eval.shape[1])] + ["y"]
        pd.DataFrame(sampled_data, columns=col_names).to_csv(
            sample_file_path, index=False
        )

        for model_name, (model_class, search_space_class) in MODEL_MAP.items():
            output_file = f"./results/{dataset}_{model_name}_{run}.csv"

            study = optuna.create_study(direction="maximize")
            study.optimize(
                lambda trial: objective(
                    trial, model_class, search_space_class, X_opt, y_opt, BATCH_SIZE
                ),
                n_trials=100,
                timeout=600,  # 10 minutes per model/dataset/run
            )

            try:
                best_params = study.best_params
                model = model_class(**best_params)
            except ValueError:
                # Triggers if all trials are pruned
                results = [{"batch": 0, "status": "ERROR: All trials pruned"}]
                pd.DataFrame(results).to_csv(output_file, index=False)
                continue

            results = []
            sampled_data_rows = []
            num_batches = len(X_eval) // BATCH_SIZE

            for b in range(num_batches):
                start_idx = b * BATCH_SIZE
                end_idx = start_idx + BATCH_SIZE
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
                    mem_bytes = asizeof.asizeof(model)
                    mc_count = model.microcluster_count()
                    status = "OK"

                except Exception as e:
                    ari, t_pred_ns, t_fit_ns, mc_count, mem_bytes = 0.0, 0, 0, 0, 0
                    status = f"ERROR: {str(e).replace(',', ';')}"

                results.append(
                    {
                        "batch": b,
                        "ari": ari,
                        "time_predict_ns": t_pred_ns,
                        "time_fit_ns": t_fit_ns,
                        "microcluster_count": mc_count,
                        "memory_bytes": mem_bytes,
                        "status": status,
                    }
                )

            pd.DataFrame(results).to_csv(output_file, index=False)
