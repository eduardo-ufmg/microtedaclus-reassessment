"""Experimental protocol shared by the main experiment and the model improvement experiments.

A task is one (dataset, model, run) triple. Its hyperparameters are tuned on the
first 20% of the stream by Bayesian optimization, and the tuned model is then
evaluated prequentially (predict, then fit) on the remaining 80%.
"""

import time
import zlib

import numpy as np
import optuna
import pandas as pd
from pympler import asizeof
from sklearn.metrics import adjusted_rand_score

from pfc_shared.previous_protocol import (
    BATCH_SIZE,
    OPT_TIMEOUT,
    OPT_TRIALS,
    SPLIT_FRACTION,
    objective,
)

BASE_SEED = 42

optuna.logging.set_verbosity(optuna.logging.WARNING)


def run_seed(dataset: str, run: int) -> int:
    """Seed of the stream and of the hyperparameter search of a (dataset, run) pair."""
    return BASE_SEED + run + (zlib.crc32(dataset.encode("utf-8")) % 10000)


def evaluate(model, X_eval, y_eval, batch_diagnostics=None) -> list[dict]:
    """Prequential evaluation in batches; stops at the first batch that fails.

    batch_diagnostics, if given, maps (y_batch, predictions) to extra columns.
    """
    results = []
    for b in range(len(X_eval) // BATCH_SIZE):
        start_idx, end_idx = b * BATCH_SIZE, (b + 1) * BATCH_SIZE
        X_batch, y_batch = X_eval[start_idx:end_idx], y_eval[start_idx:end_idx]
        extra = {}

        try:
            t0_pred_wall = time.perf_counter_ns()
            t0_pred_cpu = time.process_time_ns()
            preds = model.predict(X_batch)
            t_pred_cpu_ns = time.process_time_ns() - t0_pred_cpu
            t_pred_wall_ns = time.perf_counter_ns() - t0_pred_wall

            if batch_diagnostics is not None:
                extra = batch_diagnostics(np.asarray(y_batch), np.asarray(preds))

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
            ari = t_pred_cpu_ns = t_fit_cpu_ns = t_pred_wall_ns = t_fit_wall_ns = (
                mem_bytes
            ) = mc_count = 0
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
                **extra,
            }
        )

        if status.startswith("ERROR"):
            break  # Stop evaluation for this run constraint

    return results


def run_task(
    dataset,
    model_name,
    run,
    model_class,
    search_space_class,
    X,
    y,
    output_file,
    params_file,
    batch_diagnostics=None,
) -> dict:
    """Tune and evaluate one task, writing its result and parameter files."""
    split_idx = int(SPLIT_FRACTION * len(X))
    X_opt, y_opt = X[:split_idx], y[:split_idx]
    X_eval, y_eval = X[split_idx:], y[split_idx:]

    sampler = optuna.samplers.TPESampler(seed=run_seed(dataset, run))
    study = optuna.create_study(direction="maximize", sampler=sampler)
    started = time.perf_counter()
    study.optimize(
        lambda trial: objective(trial, model_class, search_space_class, X_opt, y_opt),
        n_trials=OPT_TRIALS,
        timeout=OPT_TIMEOUT,
    )
    summary = {
        "trials": len(study.trials),
        "tuning_seconds": time.perf_counter() - started,
    }

    try:
        best_params = study.best_params
        model = model_class(**best_params)
    except ValueError:
        pd.DataFrame([{"batch": 0, "status": "ERROR: All trials pruned"}]).to_csv(
            output_file, index=False
        )
        pd.DataFrame([]).to_csv(params_file, index=False)
        return {**summary, "status": "ERROR: All trials pruned"}

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
    pd.DataFrame(params_records).to_csv(params_file, index=False)

    results = evaluate(model, X_eval, y_eval, batch_diagnostics)
    pd.DataFrame(results).to_csv(output_file, index=False)
    return {**summary, "status": results[-1]["status"] if results else "ERROR: empty"}
