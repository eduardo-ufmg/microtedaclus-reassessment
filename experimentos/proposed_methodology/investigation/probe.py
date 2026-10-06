"""Random probe of the engine's option space on one stream (no Optuna), to see which regimes are reachable.

Usage: python investigation/probe.py DATASET [N_CONFIGS] [RUN]
"""

import os

for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMBA_NUM_THREADS",
):
    os.environ.setdefault(_var, "1")

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from investigation.cache_datasets import load_cached  # noqa: E402

EXTRA = {}


def sample_config(rng):
    return {
        "variance_limit": float(10 ** rng.uniform(-5, -1)),
        "decay": int(10 ** rng.uniform(1.5, 4)),
        "relative_limit": True,
        "membership": "nearest",
        "prune_skip": False,
        "density": "peak",
        "cap_variance": bool(rng.random() < 0.5),
        "life_mode": str(rng.choice(["reference", "recency", "recency_inactive"])),
        "activation_factor": float(rng.choice([0.0, 0.25, 0.5, 0.75, 1.0])),
        "link": float(rng.choice([1.0, 1.5, 2.0])),
        "noise_m": float(rng.choice([0.0, 2.0, 3.0, 5.0])),
        "single_assign": bool(rng.random() < 0.5),
        **EXTRA,
    }


def evaluate(args):
    dataset, run, cfg = args
    from sklearn.metrics import adjusted_rand_score

    from investigation.engine import TedaClus

    X, y = load_cached(dataset, run)
    split = int(0.2 * len(X))
    X, y = X[split:], y[split:]
    model = TedaClus(**cfg)
    aris, npred = [], []
    for b in range(len(X) // 100):
        Xb, yb = X[b * 100 : (b + 1) * 100], y[b * 100 : (b + 1) * 100]
        p = model.predict(Xb)
        model.fit(Xb)
        aris.append(adjusted_rand_score(yb, p))
        npred.append(len(np.unique(p)))
    ss = int(0.8 * len(aris))
    # the tuner's view: prequential ARI over the first 20% of the stream after the burn-in
    Xo, yo = load_cached(dataset, run)
    Xo, yo = Xo[:split], yo[:split]
    model_o = TedaClus(**cfg)
    preds = []
    for b in range(len(Xo) // 100):
        preds.append(model_o.predict(Xo[b * 100 : (b + 1) * 100]).copy())
        model_o.fit(Xo[b * 100 : (b + 1) * 100])
    preds = np.concatenate(preds)
    burn = int(0.05 * len(Xo))
    opt = adjusted_rand_score(yo[: len(preds)][burn:], preds[burn:])
    return {
        **cfg,
        "ari": float(np.mean(aris)),
        "ari_ss": float(np.mean(aris[ss:])),
        "opt": float(opt),
        "n_pred": float(np.mean(npred[ss:])),
        "mc": model._n,
        "macro": model.n_macro_,
    }


if __name__ == "__main__":
    dataset = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    run = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    tag = sys.argv[4] if len(sys.argv) > 4 else ""
    if len(sys.argv) > 5:
        import json

        EXTRA.update(json.loads(sys.argv[5]))
    rng = np.random.default_rng(123)
    tasks = [(dataset, run, sample_config(rng)) for _ in range(n)]
    with ProcessPoolExecutor(max_workers=10) as pool:
        rows = list(pool.map(evaluate, tasks, chunksize=4))
    df = pd.DataFrame(rows).sort_values("ari", ascending=False)
    out = ROOT / "results_investigation" / "_probe"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"{dataset}_{run}{tag}.csv", index=False)
    by_opt = df.sort_values("opt", ascending=False)
    print(
        f"{dataset}{tag}: spearman(tuner objective, eval ARI) = {df[['opt', 'ari']].corr(method='spearman').iloc[0, 1]:.2f} | "
        f"eval ARI of the tuner's top-1 / top-5 mean: {by_opt.ari.iloc[0]:.3f} / {by_opt.ari.head(5).mean():.3f} | best eval ARI {df.ari.max():.3f}"
    )
    pd.set_option("display.width", 250)
    cols = [
        "ari",
        "ari_ss",
        "n_pred",
        "mc",
        "macro",
        "variance_limit",
        "decay",
        "cap_variance",
        "life_mode",
        "activation_factor",
        "link",
        "noise_m",
        "single_assign",
    ]
    if "-v" in sys.argv:
        print(dataset, "top configurations of", n)
        print(df[cols].head(12).round(4).to_string(index=False))
        top = df.head(max(10, n // 10))
        print("option frequencies among the top 10%:")
        for c in [
            "cap_variance",
            "life_mode",
            "activation_factor",
            "link",
            "noise_m",
            "single_assign",
        ]:
            print("  ", c, top[c].value_counts(normalize=True).round(2).to_dict())
