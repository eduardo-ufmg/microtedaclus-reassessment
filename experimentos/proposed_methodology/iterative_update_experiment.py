#!/usr/bin/env python3
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
from sklearn.metrics import adjusted_rand_score
from tqdm import tqdm

from models.MicroTEDAClus.MicroTEDAClus import (
    MicroTEDAClus,
    MicroTEDAClusParameterSearchSpace,
)
from models.MicroTEDAClusCorrected.MicroTEDAClusCorrected_macro import (
    MicroTEDAClusCorrectedMacro,
    MicroTEDAClusCorrectedMacroParameterSearchSpace,
)
from models.MicroTEDAClusCorrected.MicroTEDAClusCorrected_orderdepend import (
    MicroTEDAClusCorrectedOrderdepend,
    MicroTEDAClusCorrectedOrderdependParameterSearchSpace,
)
from models.MicroTEDAClusCorrected.MicroTEDAClusCorrected_dimpen import (
    MicroTEDAClusCorrectedDimpen,
    MicroTEDAClusCorrectedDimpenParameterSearchSpace,
)
from models.MicroTEDAClusCorrected.MicroTEDAClusCorrected_math import (
    MicroTEDAClusCorrectedMath,
    MicroTEDAClusCorrectedMathParameterSearchSpace,
)
from models.MicroTEDAClusCorrected.MicroTEDAClusCorrected_decay import (
    MicroTEDAClusCorrectedDecay,
    MicroTEDAClusCorrectedDecayParameterSearchSpace,
)

from editorial_config import ITERATIVE_DATASET_ORDER
from iterative_config import (
    CORRECTION_STAGES,
    DEFAULT_RUNS,
    STAGE_BY_ID,
    get_stage_model_datasets,
)
from main_experiment import get_dataset

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

RUNS = DEFAULT_RUNS
BATCH_SIZE = 100
OPT_TRIALS = 64
OPT_TIMEOUT = 180
BURN_IN_FRACTION = 0.05
SPLIT_FRACTION = 0.2
BASE_SEED = 0

TAG_MODEL_MAP = {
    "orderdepend": (
        MicroTEDAClusCorrectedOrderdepend,
        MicroTEDAClusCorrectedOrderdependParameterSearchSpace,
    ),
    "dimpen": (
        MicroTEDAClusCorrectedDimpen,
        MicroTEDAClusCorrectedDimpenParameterSearchSpace,
    ),
    "math": (
        MicroTEDAClusCorrectedMath,
        MicroTEDAClusCorrectedMathParameterSearchSpace,
    ),
    "decay": (
        MicroTEDAClusCorrectedDecay,
        MicroTEDAClusCorrectedDecayParameterSearchSpace,
    ),
    "macro": (
        MicroTEDAClusCorrectedMacro,
        MicroTEDAClusCorrectedMacroParameterSearchSpace,
    ),
}

MODEL_MAP = {
    "MicroTEDAClus": (MicroTEDAClus, MicroTEDAClusParameterSearchSpace),
    "MicroTEDAClusCorrected": TAG_MODEL_MAP["macro"],
    "MicroTEDAClusCorrected_orderdepend": TAG_MODEL_MAP["orderdepend"],
    "MicroTEDAClusCorrected_dimpen": TAG_MODEL_MAP["dimpen"],
    "MicroTEDAClusCorrected_math": TAG_MODEL_MAP["math"],
    "MicroTEDAClusCorrected_decay": TAG_MODEL_MAP["decay"],
    "MicroTEDAClusCorrected_macro": TAG_MODEL_MAP["macro"],
}


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


import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Run iterative model update experiments"
    )
    parser.add_argument(
        "--stage",
        type=str,
        default=None,
        help="Specific stage ID to run (e.g. order_dependency, dimension_penalty, macrocluster_activation)",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="",
        help="Correction tag suffix (e.g. math, dimpen, orderdepend, macro)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Models to run (default: all implemented models across stages)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Datasets to run (overrides stage datasets)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=RUNS,
        help=f"Number of runs (default: {RUNS})",
    )
    args = parser.parse_args()

    canonical_model_datasets = get_stage_model_datasets()

    if args.stage:
        if args.stage not in STAGE_BY_ID:
            raise ValueError(
                f"Unknown stage: {args.stage}. Valid stages: {list(STAGE_BY_ID.keys())}"
            )
        stage = STAGE_BY_ID[args.stage]
        model_datasets = {stage.model_tag: stage.datasets}
    elif args.tag:
        tag_key = (
            f"MicroTEDAClusCorrected_{args.tag}"
            if not args.tag.startswith("MicroTEDAClus")
            else args.tag
        )
        target_ds = canonical_model_datasets.get(tag_key, ITERATIVE_DATASET_ORDER)
        model_datasets = {tag_key: target_ds}
    elif args.models:
        model_datasets = {
            m: canonical_model_datasets.get(m, ITERATIVE_DATASET_ORDER)
            for m in args.models
        }
    else:
        # Default: all models required across all defined correction stages
        model_datasets = canonical_model_datasets

    if args.datasets:
        model_datasets = {m: args.datasets for m in model_datasets}

    # Filter to only models that have an existing implementation
    implemented_model_datasets = {}
    for m, dsets in model_datasets.items():
        if m in MODEL_MAP:
            implemented_model_datasets[m] = dsets
        else:
            print(f"Notice: Model '{m}' has no registered implementation. Skipping.")

    tasks = []
    for model_name, dsets in implemented_model_datasets.items():
        for dataset in dsets:
            for run in range(args.runs):
                tasks.append((dataset, model_name, run))

    with tqdm(total=len(tasks), desc="Iterative Experiments") as pbar:
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
