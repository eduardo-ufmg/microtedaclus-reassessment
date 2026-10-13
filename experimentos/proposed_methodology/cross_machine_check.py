"""Do the stored results of the main experiment reproduce on this machine?

The models replayed here are deterministic, so running one of them with the
hyperparameters that the main experiment selected for a run, on the stream that
this machine generates for the same seed, must give the same ARI at every
batch. A run that does not reproduce was evaluated on a different stream.

Usage (needs R, like the main experiment):
    python cross_machine_check.py
Writes metrics/cross_machine/replay.csv, one row per (dataset, model, run).
"""

import os

import numpy as np
import pandas as pd

import protocol
from main_experiment import DATASETS, DIR_PARAMS, DIR_RESULTS, MODEL_MAP, RUNS
from streams import load_stream

# Model and the hyperparameters that are integers.
MODELS = {
    "MicroTEDAClusPlus": ["decay"],
    "CEDAS": ["decay_period", "min_samples"],
    "TEDACloud": [],
    "DenStream": [],
}
OUT_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "metrics", "cross_machine", "replay.csv"
)


def replay(dataset: str, model: str, run: int, X_eval, y_eval) -> dict:
    remote = pd.read_csv(os.path.join(DIR_RESULTS, f"{dataset}_{model}_{run}.csv"))
    selected = pd.read_csv(os.path.join(DIR_PARAMS, f"{dataset}_{model}_{run}.csv"))
    params = dict(zip(selected["param_name"], selected["param_value"]))
    for name in MODELS[model]:
        params[name] = int(params[name])

    local = pd.DataFrame(
        protocol.evaluate(MODEL_MAP[model][0](**params), X_eval, y_eval)
    )
    reproduced = len(local) == len(remote) and bool(
        np.max(np.abs(local["ari"].to_numpy() - remote["ari"].to_numpy())) < 1e-9
    )
    return {
        "dataset": dataset,
        "model": model,
        "run": run,
        "reproduced": reproduced,
        "stored_ari_overall": float(remote["ari"].median()),
        "replayed_ari_overall": float(local["ari"].median()),
        "stored_ari_steadystate": float(remote["ari"].iloc[-2]),
        "replayed_ari_steadystate": float(local["ari"].iloc[-2]),
    }


def main():
    rows = []
    for dataset in DATASETS:
        for run in range(RUNS):
            X, y = load_stream(dataset, run)
            split = int(protocol.SPLIT_FRACTION * len(X))
            rows += [
                replay(dataset, model, run, X[split:], y[split:]) for model in MODELS
            ]
        print(f"{dataset} done", flush=True)
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT_FILE, index=False)


if __name__ == "__main__":
    main()
