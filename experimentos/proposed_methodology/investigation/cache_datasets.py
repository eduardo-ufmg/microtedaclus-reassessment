"""Materialize every (dataset, run) stream used by experiment_runner.py as .npz files.

The cached streams are bit-identical to what experiment_runner.py feeds the models
(same seeds), so the investigation harness can run without R and in parallel.
"""

import os
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import arff

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from editorial_config import ITERATIVE_DATASET_ORDER  # noqa: E402

CACHE_DIR = ROOT / "data_samples_cache"
BASE_SEED = 0
LOCAL_ARFF = {
    "chameleon_ds3": "cluto-t7-10k.arff",
    "chameleon_ds4": "cluto-t8-8k.arff",
}


def run_seed(dataset: str, run: int) -> int:
    return BASE_SEED + run + (zlib.crc32(dataset.encode("utf-8")) % 10000)


def cache_path(dataset: str, run: int) -> Path:
    if dataset in LOCAL_ARFF:  # seed-independent
        return CACHE_DIR / f"{dataset}.npz"
    return CACHE_DIR / f"{dataset}_{run}.npz"


def load_cached(dataset: str, run: int) -> tuple[np.ndarray, np.ndarray]:
    with np.load(cache_path(dataset, run)) as f:
        return f["X"], f["y"]


def _load_arff(dataset: str) -> tuple[np.ndarray, np.ndarray]:
    data, _ = arff.loadarff(str(ROOT / "data_samples" / LOCAL_ARFF[dataset]))
    df = pd.DataFrame(data)
    target_col = df.columns[-1]
    y = df[target_col].str.decode("utf-8").replace("noise", "-1").astype(int).to_numpy()
    return df.drop(columns=[target_col]).to_numpy(), y


def main(runs: int = 32):
    CACHE_DIR.mkdir(exist_ok=True)
    get_dataset = None
    for dataset in ITERATIVE_DATASET_ORDER:
        for run in range(runs):
            path = cache_path(dataset, run)
            if path.exists():
                continue
            if dataset in LOCAL_ARFF:
                X, y = _load_arff(dataset)
            else:
                if get_dataset is None:
                    from main_experiment import get_dataset
                X, y = get_dataset(dataset, run_seed(dataset, run))
            np.savez_compressed(
                path, X=np.asarray(X, dtype=float), y=np.asarray(y, dtype=int)
            )
        print(f"cached {dataset}", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 32)
