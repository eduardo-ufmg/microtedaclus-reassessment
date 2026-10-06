"""Check that MicroTEDAClusVectorized reproduces the stored reference results."""

import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from editorial_config import ITERATIVE_DATASET_ORDER  # noqa: E402
from investigation.cache_datasets import load_cached  # noqa: E402
from models.MicroTEDAClusVectorized.MicroTEDAClusVectorized import (  # noqa: E402
    MicroTEDAClusVectorized,
)

BATCH_SIZE = 100
SPLIT_FRACTION = 0.2


def check(task):
    dataset, run = task
    res_file = ROOT / "results_experiments" / f"{dataset}_MicroTEDAClus_{run}.csv"
    par_file = (
        ROOT / "results_params_experiments" / f"{dataset}_MicroTEDAClus_{run}.csv"
    )
    if not res_file.exists() or not par_file.exists():
        return None
    ref = pd.read_csv(res_file)
    par = pd.read_csv(par_file)
    if par.empty or "ari" not in ref.columns:
        return None
    params = dict(zip(par["param_name"], par["param_value"]))
    params["decay"] = int(params["decay"])

    X, y = load_cached(dataset, run)
    split = int(SPLIT_FRACTION * len(X))
    X, y = X[split:], y[split:]

    model = MicroTEDAClusVectorized(**params)
    aris, mcs = [], []
    t0 = time.process_time()
    for b in range(len(X) // BATCH_SIZE):
        Xb, yb = (
            X[b * BATCH_SIZE : (b + 1) * BATCH_SIZE],
            y[b * BATCH_SIZE : (b + 1) * BATCH_SIZE],
        )
        preds = model.predict(Xb)
        model.fit(Xb)
        aris.append(adjusted_rand_score(yb, preds))
        mcs.append(model.microcluster_count())
    elapsed = time.process_time() - t0

    ref_time = (ref["time_predict_cpu_ns"].sum() + ref["time_fit_cpu_ns"].sum()) / 1e9
    return {
        "dataset": dataset,
        "run": run,
        "ari_equal": bool(np.allclose(aris, ref["ari"].to_numpy(), atol=1e-12)),
        "mc_equal": bool(np.array_equal(mcs, ref["microcluster_count"].to_numpy())),
        "time_ref_s": ref_time,
        "time_vec_s": elapsed,
    }


def main():
    tasks = [(d, r) for d in ITERATIVE_DATASET_ORDER for r in range(32)]
    with ProcessPoolExecutor(max_workers=10) as pool:
        rows = [r for r in pool.map(check, tasks, chunksize=4) if r is not None]
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "investigation" / "validate_vectorized.csv", index=False)
    summary = df.groupby("dataset", sort=False).agg(
        runs=("run", "count"),
        ari_equal=("ari_equal", "sum"),
        mc_equal=("mc_equal", "sum"),
        time_ref_s=("time_ref_s", "median"),
        time_vec_s=("time_vec_s", "median"),
    )
    summary["speedup"] = (
        df.assign(s=df.time_ref_s / df.time_vec_s)
        .groupby("dataset", sort=False)
        .s.median()
    )
    pd.set_option("display.width", 200)
    print(summary.round(3))
    print(df[~(df.ari_equal & df.mc_equal)].to_string(index=False))


if __name__ == "__main__":
    main()
