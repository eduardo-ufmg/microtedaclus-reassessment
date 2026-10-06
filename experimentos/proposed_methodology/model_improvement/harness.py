"""Parallel runner of the model improvement experiments.

Every (variant, dataset, run) task goes through protocol.run_task, the same
procedure the main experiment applies to its models, on streams read from the
cache. The result files therefore have the columns of the main experiment, plus
a few per-batch diagnostics of the predicted partition.

Usage (from experimentos/proposed_methodology):
    python -m model_improvement.harness --all
    python -m model_improvement.harness --experiment corrections_single [...]
    python -m model_improvement.harness --variants final --first-run 126 --runs 94
"""

import argparse
import json
import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import homogeneity_completeness_v_measure
from threadpoolctl import threadpool_limits

import protocol
from model_improvement.experiments import (
    DATASETS,
    DEVELOPMENT_RUNS,
    EXPERIMENT_BY_KEY,
    EXPERIMENTS,
)
from model_improvement.variants import resolve
from streams import load_stream, missing_from_cache

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results_model_improvement"

# Longest streams first, so the pool stays balanced
STREAM_WEIGHT = {"chameleon_ds3": 3, "random_rbf_events": 3, "chameleon_ds4": 2}


def result_paths(variant: str, dataset: str, run: int) -> tuple[Path, Path]:
    """Result and parameter files of a task."""
    base = RESULTS_DIR / variant
    return base / f"{dataset}_{run}.csv", base / f"{dataset}_{run}.params.csv"


def batch_diagnostics(y_batch: np.ndarray, preds: np.ndarray) -> dict:
    """Shape of the predicted partition of a batch."""
    homogeneity, completeness, _ = homogeneity_completeness_v_measure(y_batch, preds)
    return {
        "n_true": len(np.unique(y_batch)),
        "n_pred": len(np.unique(preds)),
        "homogeneity": homogeneity,
        "completeness": completeness,
        "noise_pred": float(np.mean(preds == -1)),
    }


def run_task(task: tuple) -> dict:
    variant, dataset, run, force = task
    info = {"variant": variant, "dataset": dataset, "run": run}
    output_file, params_file = result_paths(variant, dataset, run)
    if output_file.exists() and params_file.exists() and not force:
        return {**info, "status": "cached"}

    model_class, search_space_class = resolve(variant)
    X, y = load_stream(dataset, run)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    summary = protocol.run_task(
        dataset,
        variant,
        run,
        model_class,
        search_space_class,
        X,
        y,
        output_file,
        params_file,
        batch_diagnostics,
    )
    return {**info, **summary}


def _single_threaded():
    """Each worker uses one BLAS/OpenMP thread: the pool already occupies every core."""
    threadpool_limits(limits=1)


def run_block(variants, runs, datasets=None, workers=None, force=False) -> pd.DataFrame:
    """Run every (variant, dataset, run) task of a block; returns one row per task."""
    datasets = datasets or DATASETS
    missing = missing_from_cache(datasets, runs)
    if missing:
        raise SystemExit(
            f"{len(missing)} streams are not cached (first: {missing[0]}); "
            f"run `python -m streams {runs.start} {runs.stop - 1}`"
        )

    tasks = [(v, d, r, force) for v in variants for d in datasets for r in runs]
    tasks.sort(key=lambda t: -STREAM_WEIGHT.get(t[1], 1))
    workers = workers or max(1, (os.cpu_count() or 2) - 2)
    with ProcessPoolExecutor(max_workers=workers, initializer=_single_threaded) as pool:
        rows = [
            f.result() for f in as_completed([pool.submit(run_task, t) for t in tasks])
        ]
    return pd.DataFrame(rows)


def code_state() -> dict:
    """Commit of the code and whether the experiment code has uncommitted changes."""

    def git(*args):
        return subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True
        ).stdout.strip()

    dirty = git(
        "status",
        "--porcelain",
        "--",
        "model_improvement",
        "models",
        "protocol.py",
        "streams.py",
    )
    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(dirty)}


def record(
    block: str,
    variants,
    runs,
    datasets,
    tasks: pd.DataFrame,
    seconds: float,
    state: dict,
):
    """Append the block to the manifest and its tuning statistics to the tuning log."""
    RESULTS_DIR.mkdir(exist_ok=True)
    manifest = RESULTS_DIR / "MANIFEST.json"
    history = json.loads(manifest.read_text()) if manifest.exists() else []
    history.append(
        {
            "block": block,
            "variants": list(variants),
            "runs": [runs.start, runs.stop - 1],
            "datasets": list(datasets),
            "statuses": tasks["status"].value_counts().to_dict(),
            "seconds": round(seconds),
            "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
            "opt_trials": protocol.OPT_TRIALS,
            "opt_timeout": protocol.OPT_TIMEOUT,
            "base_seed": protocol.BASE_SEED,
            **state,
        }
    )
    manifest.write_text(json.dumps(history, indent=1))

    executed = tasks[tasks["status"] != "cached"]
    if not executed.empty:
        log = RESULTS_DIR / "tuning.csv"
        executed.to_csv(log, mode="a", header=not log.exists(), index=False)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="every experiment, in the order of the chapter",
    )
    parser.add_argument(
        "--experiment", nargs="+", default=[], choices=list(EXPERIMENT_BY_KEY)
    )
    parser.add_argument("--variants", nargs="+", default=[])
    parser.add_argument("--first-run", type=int, default=DEVELOPMENT_RUNS.start)
    parser.add_argument("--runs", type=int, default=len(DEVELOPMENT_RUNS))
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="run with uncommitted changes to the experiment code",
    )
    args = parser.parse_args()

    state = code_state()
    if state["dirty"] and not args.allow_dirty:
        raise SystemExit(
            "the experiment code has uncommitted changes; commit them or pass --allow-dirty"
        )

    blocks = [
        (e.key, e.variants, e.runs)
        for e in (
            EXPERIMENTS if args.all else map(EXPERIMENT_BY_KEY.get, args.experiment)
        )
    ]
    if args.variants:
        blocks.append(
            ("custom", args.variants, range(args.first_run, args.first_run + args.runs))
        )
    if not blocks:
        parser.error("nothing to run: pass --all, --experiment or --variants")

    for name, variants, runs in blocks:
        started = time.time()
        tasks = run_block(variants, runs, args.datasets, args.workers, args.force)
        seconds = time.time() - started
        record(name, variants, runs, args.datasets or DATASETS, tasks, seconds, state)
        print(
            f"{name}: {tasks['status'].value_counts().to_dict()} in {seconds:.0f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
