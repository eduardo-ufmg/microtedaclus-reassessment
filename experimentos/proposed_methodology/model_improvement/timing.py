"""Serial timing of the models that matter.

The experiments run many tasks in parallel, which disturbs wall-clock times.
Here each model replays the evaluation segment of a few development runs with
the hyperparameters tuned for it in that run, one task at a time on an
otherwise idle machine. Compiled kernels are warmed up before being timed.

Usage (from experimentos/proposed_methodology):
    python -m model_improvement.timing [RUNS_PER_DATASET]
"""

import sys

import pandas as pd

import protocol
from model_improvement.experiments import DATASETS, DEVELOPMENT_RUNS
from model_improvement.harness import RESULTS_DIR, result_paths
from model_improvement.variants import resolve
from streams import load_stream

MODELS = ["original", "vectorized", "corrected", "improved", "final"]
OUT_FILE = RESULTS_DIR / "_timing" / "timing.csv"


def tuned_parameters(variant: str, dataset: str, run: int) -> dict | None:
    params_file = result_paths(variant, dataset, run)[1]
    if not params_file.exists():
        return None
    params = pd.read_csv(params_file)
    if params.empty:
        return None
    return dict(zip(params["param_name"], params["param_value"]))


def main(runs_per_dataset: int = 8):
    rows = []
    for dataset in DATASETS:
        for run in list(DEVELOPMENT_RUNS)[:runs_per_dataset]:
            X, y = load_stream(dataset, run)
            split = int(protocol.SPLIT_FRACTION * len(X))
            X_eval, y_eval = X[split:], y[split:]
            for variant in MODELS:
                params = tuned_parameters(variant, dataset, run)
                if params is None:
                    continue
                model_class, _ = resolve(variant)
                protocol.evaluate(
                    model_class(**params), X_eval[:300], y_eval[:300]
                )  # warm-up
                results = pd.DataFrame(
                    protocol.evaluate(model_class(**params), X_eval, y_eval)
                )
                rows.append(
                    {
                        "dataset": dataset,
                        "run": run,
                        "model": variant,
                        "time_fit_ms": results["time_fit_wall_ns"].sum() / 1e6,
                        "time_predict_ms": results["time_predict_wall_ns"].sum() / 1e6,
                        "microclusters": results["microcluster_count"].iloc[-2],
                    }
                )
        print(f"timed {dataset}", flush=True)
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT_FILE, index=False)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 8)
