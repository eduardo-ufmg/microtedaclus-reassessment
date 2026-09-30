#!/usr/bin/env python3
"""Unified experiment runner for streaming clustering models."""

import argparse
import logging
import os
import sys
import time
import zlib

import numpy as np
import optuna
import pandas as pd
import rpy2.rinterface_lib.callbacks
import rpy2.robjects as robjects
from pympler import asizeof
from rpy2.robjects.packages import importr
from sklearn.metrics import adjusted_rand_score
from tqdm import tqdm

from editorial_config import ITERATIVE_DATASET_ORDER
from main_experiment import get_dataset
from stages_config import (
    ALL_STAGES,
    DEFAULT_RUNS,
    STAGE_BY_ID,
    get_model_registry,
)

rpy2.rinterface_lib.callbacks.logger.setLevel(logging.ERROR)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Load R streaming dependencies
try:
    stream = importr("stream")
    streamMOA = importr("streamMOA")
    mlbench = importr("mlbench")
except Exception as e:
    print(f"Critical Error loading R dependencies via rpy2: {e}")
    sys.exit(1)

BATCH_SIZE = 100
OPT_TRIALS = 64
OPT_TIMEOUT = 180
BURN_IN_FRACTION = 0.05
SPLIT_FRACTION = 0.2
BASE_SEED = 0


def objective(trial, model_class, search_space_class, X_opt, y_opt):
    """Optuna objective optimizing prequential ARI penalized by execution time."""
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


def main():
    parser = argparse.ArgumentParser(
        description="Unified streaming clustering experiment runner"
    )
    parser.add_argument(
        "--stage",
        type=str,
        default=None,
        help="Specific stage ID to run (e.g. order_dependency, dimension_penalty, base_improved)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all defined stages",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Explicit models to run",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Explicit datasets to run",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=DEFAULT_RUNS,
        help=f"Number of randomized repetitions (default: {DEFAULT_RUNS})",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="./results_experiments",
        help="Directory to save raw batch CSVs (default: ./results_experiments)",
    )
    parser.add_argument(
        "--params-dir",
        type=str,
        default="./results_params_experiments",
        help="Directory to save optimal parameters (default: ./results_params_experiments)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve and list tasks without executing them",
    )
    parser.add_argument(
        "--opt-trials",
        type=int,
        default=OPT_TRIALS,
        help=f"Number of Optuna optimization trials (default: {OPT_TRIALS})",
    )
    parser.add_argument(
        "--opt-timeout",
        type=int,
        default=OPT_TIMEOUT,
        help=f"Optuna timeout per task in seconds (default: {OPT_TIMEOUT})",
    )
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.params_dir, exist_ok=True)

    model_registry = get_model_registry()

    # Determine required (model, datasets) mappings
    if args.stage:
        if args.stage not in STAGE_BY_ID:
            raise ValueError(
                f"Unknown stage '{args.stage}'. Valid stages: {list(STAGE_BY_ID.keys())}"
            )
        stage = STAGE_BY_ID[args.stage]
        model_datasets = {m: stage.datasets for m in stage.models}
    elif args.all:
        model_datasets = {}
        for stage in ALL_STAGES:
            for m in stage.models:
                if m not in model_datasets:
                    model_datasets[m] = stage.datasets
    elif args.models:
        model_datasets = {
            m: args.datasets if args.datasets else ITERATIVE_DATASET_ORDER
            for m in args.models
        }
    else:
        # Default: all models from all defined stages
        model_datasets = {}
        for stage in ALL_STAGES:
            for m in stage.models:
                if m not in model_datasets:
                    model_datasets[m] = stage.datasets

    if args.datasets and not args.models:
        model_datasets = {m: args.datasets for m in model_datasets}

    # Filter to implemented models only
    implemented_model_datasets = {}
    for m, dsets in model_datasets.items():
        if m in model_registry:
            implemented_model_datasets[m] = dsets
        else:
            print(
                f"Notice: Model '{m}' is not yet implemented or registered. Skipping."
            )

    # Generate deduplicated (dataset, model, run) tasks
    tasks = []
    seen = set()
    for model_name, dsets in implemented_model_datasets.items():
        for dataset in dsets:
            for run in range(args.runs):
                task_key = (dataset, model_name, run)
                if task_key not in seen:
                    seen.add(task_key)
                    tasks.append(task_key)

    if args.dry_run:
        print(f"\n[Dry Run] Total tasks resolved: {len(tasks)}")
        for m, dsets in implemented_model_datasets.items():
            print(
                f"  Model: {m:35s} | Datasets: {len(dsets):2d} | Runs: {args.runs:2d} -> {len(dsets) * args.runs} tasks"
            )
        return

    print(
        f"\nStarting execution of {len(tasks)} tasks across {len(implemented_model_datasets)} models."
    )

    with tqdm(total=len(tasks), desc="Experiments") as pbar:
        for dataset, model_name, run in tasks:
            output_file = os.path.join(
                args.results_dir, f"{dataset}_{model_name}_{run}.csv"
            )
            params_file = os.path.join(
                args.params_dir, f"{dataset}_{model_name}_{run}.csv"
            )

            if os.path.exists(output_file) and os.path.exists(params_file):
                pbar.update(1)
                continue

            dataset_hash = zlib.crc32(dataset.encode("utf-8"))
            run_seed = BASE_SEED + run + (dataset_hash % 10000)
            X, y = get_dataset(dataset, run_seed)
            split_idx = int(SPLIT_FRACTION * len(X))
            X_opt, y_opt = X[:split_idx], y[:split_idx]
            X_eval, y_eval = X[split_idx:], y[split_idx:]

            model_class, search_space_class = model_registry[model_name]
            sampler = optuna.samplers.TPESampler(seed=run_seed)
            study = optuna.create_study(direction="maximize", sampler=sampler)
            study.optimize(
                lambda trial: objective(
                    trial, model_class, search_space_class, X_opt, y_opt
                ),
                n_trials=args.opt_trials,
                timeout=args.opt_timeout,
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
