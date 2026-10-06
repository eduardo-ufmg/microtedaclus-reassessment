"""Same-machine timing of the reference and vectorized implementations at identical hyperparameters."""

import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from editorial_config import ITERATIVE_DATASET_ORDER  # noqa: E402
from investigation.cache_datasets import load_cached  # noqa: E402

RUNS = 4


def time_dataset(dataset):
    from models.MicroTEDAClus.MicroTEDAClus import MicroTEDAClus
    from models.MicroTEDAClusVectorized.MicroTEDAClusVectorized import (
        MicroTEDAClusVectorized,
    )

    rows = []
    for run in range(RUNS):
        par = pd.read_csv(
            ROOT / "results_params_experiments" / f"{dataset}_MicroTEDAClus_{run}.csv"
        )
        params = dict(zip(par["param_name"], par["param_value"]))
        params["decay"] = int(params["decay"])
        X, _ = load_cached(dataset, run)
        X = X[int(0.2 * len(X)) :]
        for name, cls in (
            ("vectorized", MicroTEDAClusVectorized),
            ("reference", MicroTEDAClus),
        ):
            for rep in range(
                2 if name == "vectorized" else 1
            ):  # first vectorized pass warms the JIT cache
                model = cls(**params)
                fit = pred = 0
                for b in range(len(X) // 100):
                    Xb = X[b * 100 : (b + 1) * 100]
                    t0 = time.process_time_ns()
                    model.predict(Xb)
                    t1 = time.process_time_ns()
                    model.fit(Xb)
                    t2 = time.process_time_ns()
                    pred += t1 - t0
                    fit += t2 - t1
            rows.append(
                {
                    "dataset": dataset,
                    "run": run,
                    "model": name,
                    "fit_ms": fit / 1e6,
                    "predict_ms": pred / 1e6,
                    "mc": model.microcluster_count(),
                }
            )
    return rows


if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=4) as pool:
        rows = [r for rs in pool.map(time_dataset, ITERATIVE_DATASET_ORDER) for r in rs]
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "investigation" / "time_reference.csv", index=False)
    piv = df.pivot_table(
        index="dataset",
        columns="model",
        values=["fit_ms", "predict_ms"],
        aggfunc="median",
    ).reindex(ITERATIVE_DATASET_ORDER)
    piv[("speedup", "fit")] = (
        piv[("fit_ms", "reference")] / piv[("fit_ms", "vectorized")]
    )
    piv[("speedup", "predict")] = (
        piv[("predict_ms", "reference")] / piv[("predict_ms", "vectorized")]
    )
    pd.set_option("display.width", 200)
    print(piv.round(2))
