"""Serial timing of the models that matter.

The experiments run many tasks in parallel, which disturbs wall-clock times.
Here each model replays the evaluation segment of the development runs, one
task at a time on an otherwise idle machine. Compiled kernels are warmed up
before being timed.

Two questions are answered, and the column `hyperparameters` of the output
tells them apart:

- the cost of each model as it is used, with the hyperparameters tuned for it
  in that run (`hyperparameters` equal to `model`);
- the speedup of the vectorized implementation over the original one, with
  both running the same configuration: each of the two is also timed with the
  hyperparameters tuned for the other.

The whole pass is repeated and every repetition is kept (column `repetition`):
the time of a task is the minimum over its repetitions, which discards the
slowdowns that the machine, and not the model, causes. On a processor with
cores of different speeds the process must also be restricted to cores of one
category, as in the usage line below.

Usage (from experimentos/proposed_methodology):
    taskset -c PERFORMANCE_CORES python -m model_improvement.timing [RUNS_PER_DATASET] [REPETITIONS]
"""

import sys

import pandas as pd

import protocol
from model_improvement.experiments import DATASETS, DEVELOPMENT_RUNS
from model_improvement.harness import RESULTS_DIR, result_paths
from model_improvement.variants import resolve
from streams import load_stream

MODELS = ["original", "vectorized", "c_nearest", "ic_single_noise_relative", "ir_nopen"]
# (model, variant whose tuned hyperparameters it runs with)
MEASUREMENTS = [(model, model) for model in MODELS] + [
    ("original", "vectorized"),
    ("vectorized", "original"),
]
WARM_UP_SAMPLES = 300
OUT_FILE = RESULTS_DIR / "_timing" / "timing.csv"


def tuned_parameters(variant: str, dataset: str, run: int) -> dict | None:
    params_file = result_paths(variant, dataset, run)[1]
    if not params_file.exists():
        return None
    params = pd.read_csv(params_file)
    if params.empty:
        return None
    return dict(zip(params["param_name"], params["param_value"]))


def time_model(model: str, params: dict, X_eval, y_eval) -> dict:
    """Time one replay of the evaluation segment, after warming up the model's kernels."""
    model_class, _ = resolve(model)
    protocol.evaluate(
        model_class(**params), X_eval[:WARM_UP_SAMPLES], y_eval[:WARM_UP_SAMPLES]
    )
    results = pd.DataFrame(protocol.evaluate(model_class(**params), X_eval, y_eval))
    return {
        "time_fit_ms": results["time_fit_wall_ns"].sum() / 1e6,
        "time_predict_ms": results["time_predict_wall_ns"].sum() / 1e6,
        "microclusters": results["microcluster_count"].iloc[-2],
    }


def timing_pass(runs_per_dataset: int) -> list[dict]:
    rows = []
    for dataset in DATASETS:
        for run in list(DEVELOPMENT_RUNS)[:runs_per_dataset]:
            X, y = load_stream(dataset, run)
            split = int(protocol.SPLIT_FRACTION * len(X))
            X_eval, y_eval = X[split:], y[split:]
            for model, source in MEASUREMENTS:
                params = tuned_parameters(source, dataset, run)
                if params is None:
                    continue
                rows.append(
                    {
                        "dataset": dataset,
                        "run": run,
                        "model": model,
                        "hyperparameters": source,
                        **time_model(model, params, X_eval, y_eval),
                    }
                )
    return rows


def main(runs_per_dataset: int = len(DEVELOPMENT_RUNS), repetitions: int = 3):
    rows = []
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    for repetition in range(repetitions):
        rows += [
            {**row, "repetition": repetition} for row in timing_pass(runs_per_dataset)
        ]
        pd.DataFrame(rows).to_csv(OUT_FILE, index=False)
        print(f"repetition {repetition + 1} of {repetitions} done", flush=True)


if __name__ == "__main__":
    main(*map(int, sys.argv[1:3]))
