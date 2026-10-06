"""Parallel experiment harness mirroring experiment_runner.py on cached streams.

Protocol (identical to experiment_runner.py): per (dataset, run), the first 20%
of the stream tunes the hyperparameters with Optuna (TPE, 64 trials, 180 s,
prequential ARI after a 5% burn-in minus 1e-3 * seconds); the remaining 80% is
evaluated prequentially in batches of 100 (predict, then fit).

Usage:
    python investigation/harness.py VARIANT [VARIANT ...] [--runs N] [--datasets ...]
"""

import argparse
import os

# One BLAS/OpenMP thread per worker: the pool already uses every core
for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMBA_NUM_THREADS",
):
    os.environ.setdefault(_var, "1")

import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from editorial_config import ITERATIVE_DATASET_ORDER  # noqa: E402
from investigation.cache_datasets import load_cached, run_seed  # noqa: E402

BATCH_SIZE = 100
OPT_TRIALS = 64
OPT_TIMEOUT = 180
BURN_IN_FRACTION = 0.05
SPLIT_FRACTION = 0.2

RESULTS_DIR = ROOT / "results_investigation"


def _prequential(model, X, y, record=False):
    from sklearn.metrics import adjusted_rand_score, homogeneity_completeness_v_measure

    rows, preds_all = [], []
    for b in range(len(X) // BATCH_SIZE):
        Xb, yb = (
            X[b * BATCH_SIZE : (b + 1) * BATCH_SIZE],
            y[b * BATCH_SIZE : (b + 1) * BATCH_SIZE],
        )
        t0 = time.process_time_ns()
        preds = np.asarray(model.predict(Xb))
        t1 = time.process_time_ns()
        model.fit(Xb)
        t2 = time.process_time_ns()
        preds_all.append(preds.copy())
        if record:
            hom, comp, _ = homogeneity_completeness_v_measure(yb, preds)
            rows.append(
                {
                    "batch": b,
                    "ari": adjusted_rand_score(yb, preds),
                    "time_predict_cpu_ns": t1 - t0,
                    "time_fit_cpu_ns": t2 - t1,
                    "microcluster_count": model.microcluster_count(),
                    "n_pred": len(np.unique(preds)),
                    "n_true": len(np.unique(yb)),
                    "homogeneity": hom,
                    "completeness": comp,
                    "noise_pred": float(np.mean(preds == -1)),
                }
            )
    return rows, preds_all


def run_task(task):
    variant, dataset, run, opt_trials, opt_timeout, force = task
    out_dir = RESULTS_DIR / variant
    out_file = out_dir / f"{dataset}_{run}.csv"
    par_file = out_dir / f"{dataset}_{run}.params.csv"
    if out_file.exists() and par_file.exists() and not force:
        return task[:3], "cached"

    import optuna
    from sklearn.metrics import adjusted_rand_score

    from investigation.variants import resolve

    warnings.filterwarnings("ignore")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    model_class, space_class, fixed = resolve(variant)

    X, y = load_cached(dataset, run)
    split = int(SPLIT_FRACTION * len(X))
    X_opt, y_opt, X_eval, y_eval = X[:split], y[:split], X[split:], y[split:]
    burn_in = int(BURN_IN_FRACTION * len(X_opt))

    def objective(trial):
        params = space_class().sample_parameters(trial)
        try:
            model = model_class(**fixed, **params)
            t0 = time.process_time_ns()
            _, preds = _prequential(model, X_opt, y_opt)
            seconds = (time.process_time_ns() - t0) / 1e9
        except Exception:
            raise optuna.exceptions.TrialPruned()
        preds = np.concatenate(preds)
        return (
            adjusted_rand_score(y_opt[: len(preds)][burn_in:], preds[burn_in:])
            - 1e-3 * seconds
        )

    seed = run_seed(dataset, run)
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed)
    )
    t_opt = time.time()
    study.optimize(objective, n_trials=opt_trials, timeout=opt_timeout)
    t_opt = time.time() - t_opt

    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        best = study.best_params
    except ValueError:
        pd.DataFrame([{"batch": 0, "status": "ERROR: all trials pruned"}]).to_csv(
            out_file, index=False
        )
        pd.DataFrame([]).to_csv(par_file, index=False)
        return task[:3], "all-pruned"

    try:
        model = model_class(**fixed, **best)
        rows, _ = _prequential(model, X_eval, y_eval, record=True)
        status = "ok"
    except Exception as e:  # keep the failure visible in the result file
        rows, status = [{"batch": 0, "ari": 0.0, "status": f"ERROR: {e!r}"}], "error"
    pd.DataFrame(rows).to_csv(out_file, index=False)
    pd.DataFrame(
        [{"param_name": k, "param_value": v} for k, v in best.items()]
        + [
            {"param_name": "_opt_trials", "param_value": len(study.trials)},
            {"param_name": "_opt_seconds", "param_value": t_opt},
        ]
    ).to_csv(par_file, index=False)
    return task[:3], status


def run_variants(
    variants,
    datasets=None,
    runs=32,
    workers=None,
    opt_trials=OPT_TRIALS,
    opt_timeout=OPT_TIMEOUT,
    force=False,
    quiet=False,
    first_run=0,
):
    datasets = datasets or list(ITERATIVE_DATASET_ORDER)
    workers = workers or max(1, (os.cpu_count() or 2) - 2)
    tasks = [
        (v, d, r, opt_trials, opt_timeout, force)
        for v in variants
        for d in datasets
        for r in range(first_run, first_run + runs)
    ]
    # Longest streams first so the pool stays balanced
    tasks.sort(
        key=lambda t: -{
            "chameleon_ds3": 3,
            "random_rbf_events": 3,
            "chameleon_ds4": 2,
        }.get(t[1], 1)
    )
    t0 = time.time()
    counts = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_task, t) for t in tasks]
        for f in as_completed(futures):
            _, status = f.result()
            counts[status] = counts.get(status, 0) + 1
    if not quiet:
        print(f"{variants}: {counts} in {time.time() - t0:.0f}s", flush=True)
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("variants", nargs="+")
    parser.add_argument("--runs", type=int, default=32)
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--first-run",
        type=int,
        default=0,
        help="index of the first run (32 selects the confirmation seeds)",
    )
    args = parser.parse_args()
    run_variants(
        args.variants,
        args.datasets,
        args.runs,
        args.workers,
        force=args.force,
        first_run=args.first_run,
    )


if __name__ == "__main__":
    main()
