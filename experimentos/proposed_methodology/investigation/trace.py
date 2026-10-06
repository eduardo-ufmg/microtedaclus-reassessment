"""Replay one tuned (variant, dataset, run) task and print the evolution of the model state.

Usage: python investigation/trace.py VARIANT DATASET RUN [STEP]
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from investigation.cache_datasets import load_cached  # noqa: E402
from investigation.variants import resolve  # noqa: E402


def tuned_params(variant, dataset, run):
    par = pd.read_csv(
        ROOT / "results_investigation" / variant / f"{dataset}_{run}.params.csv"
    )
    par = par[~par.param_name.str.startswith("_")]
    params = dict(zip(par.param_name, par.param_value))
    for k in ("decay",):
        if k in params:
            params[k] = int(float(params[k]))
    return {
        k: (
            float(v)
            if isinstance(v, str)
            and v.replace(".", "", 1).replace("e-", "", 1).isdigit()
            else v
        )
        for k, v in params.items()
    }


def trace(variant, dataset, run, step=4, params=None):
    cls, _, fixed = resolve(variant)
    params = params or tuned_params(variant, dataset, run)
    print(
        variant,
        dataset,
        run,
        {k: (round(v, 5) if isinstance(v, float) else v) for k, v in params.items()},
    )
    X, y = load_cached(dataset, run)
    split = int(0.2 * len(X))
    X, y = X[split:], y[split:]
    model = cls(**fixed, **params)
    print(
        " batch   ari  true pred  noise%   mc  mature active macro   med_S  max_S  med_sd  max_sd"
    )
    for b in range(len(X) // 100):
        Xb, yb = X[b * 100 : (b + 1) * 100], y[b * 100 : (b + 1) * 100]
        p = model.predict(Xb)
        if b % step == 0 and hasattr(model, "_arrays") and model._n > 0:
            mass, var, act = (
                model._view("mass"),
                model._view("variance"),
                model._view("active"),
            )
            a = act & (mass > 2)
            ms = mass[a] if a.any() else np.zeros(1)
            sd = np.sqrt(var[a]) if a.any() else np.zeros(1)
            print(
                f"{b:6d} {adjusted_rand_score(yb, p):5.2f} {len(np.unique(yb)):5d} {len(np.unique(p)):4d} {100 * np.mean(p == -1):6.0f} "
                f"{model._n:5d} {int((mass > 2).sum()):6d} {int(a.sum()):6d} {model.n_macro_:5d} {np.median(ms):7.0f} {ms.max():6.0f} {np.median(sd):7.3f} {sd.max():7.3f}"
            )
        model.fit(Xb)
    return model


if __name__ == "__main__":
    trace(
        sys.argv[1],
        sys.argv[2],
        int(sys.argv[3]),
        int(sys.argv[4]) if len(sys.argv) > 4 else 4,
    )
