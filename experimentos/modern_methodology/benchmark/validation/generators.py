"""Validation of the ported generators against the streams that R generated.

For the four datasets that the previous experiment generated with R, the
ported generator is compared with the R-generated streams in three ways:

- distribution: for the generators whose samples are independent given the
  time (Cassini and the two benchmarks), two-sample tests on the pooled
  coordinates of every cluster and on the cluster proportions;
- descriptors: statistics of the structure of a stream, computed in windows
  and summarized per stream, which also cover the stream whose samples depend
  on one another (Random RBF with events);
- behaviour: the ARI of models with fixed hyperparameters on both sources.

Usage (from experimentos/modern_methodology):
    python -m benchmark.validation.generators
The R-generated streams are the cache of the previous experiment
(experimentos/proposed_methodology/data_samples_cache, runs 94 to 219), which
`python -m streams 94 219` fills there, with R.
"""

import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp
from statsmodels.stats.multitest import multipletests

from benchmark.datasets import load_stream
from benchmark.evaluation import ari_summary, prequential
from benchmark.models.microtedaclus_plus import MicroTEDAClusPlus
from benchmark.models.scikit import Birch, MiniBatchKMeans
from pfc_shared.statistics import (
    ALPHA,
    CLIFFS_DELTA_THRESHOLD,
    cliffs_delta,
    safe_mannwhitney,
)

ROOT = Path(__file__).resolve().parents[2]
REFERENCE_DIR = ROOT.parent / "proposed_methodology" / "data_samples_cache"
OUT_DIR = ROOT / "results_validation" / "generators"

DATASETS = ["cassini", "benchmark_1", "benchmark_2", "random_rbf_events"]
INDEPENDENT_SAMPLES = ["cassini", "benchmark_1", "benchmark_2"]
REFERENCE_RUNS = range(
    94, 220
)  # runs of the previous experiment that were generated here with R
PORT_RUNS = range(len(REFERENCE_RUNS))
WINDOW = 500
BLOCK = 1000
INITIAL_RBF_CLUSTERS = 3
DESCRIPTORS = ["clusters", "noise", "spread", "displacement", "separation"]
# Number of clusters at the start of each stream, for the models that need it.
STARTING_CLUSTERS = {
    "cassini": 3,
    "benchmark_1": 2,
    "benchmark_2": 3,
    "random_rbf_events": 3,
}
MODELS = {
    "MicroTEDAClus+": lambda k: MicroTEDAClusPlus(),
    "mini-batch k-means": lambda k: MiniBatchKMeans(n_clusters=k),
    "BIRCH": lambda k: Birch(threshold=0.05, n_clusters=k),
}


def reference_stream(dataset: str, run: int):
    """Stream that R generated for a run of the previous experiment; clusters from 0, noise -1."""
    seed = 42 + run + zlib.crc32(dataset.encode("utf-8")) % 10000
    with np.load(REFERENCE_DIR / f"{dataset}_seed{seed}.npz") as stored:
        X, y = stored["X"], stored["y"]
    return X, np.where(y == -1, -1, y - 1)


def with_reference_labels(y: np.ndarray) -> np.ndarray:
    """Labels as the R wrapper gives them: the first cluster created after the initial ones is noise."""
    return np.where(y == INITIAL_RBF_CLUSTERS, -1, y)


def sources(dataset: str):
    """(source, run, X, y) of every stream of a dataset."""
    for run in REFERENCE_RUNS:
        yield "R", run, *reference_stream(dataset, run)
    for run in PORT_RUNS:
        X, y = load_stream(dataset, run)
        if dataset == "random_rbf_events":
            yield "port", run, X, with_reference_labels(y)
            yield "port, own labels", run, X, y
        else:
            yield "port", run, X, y


def window_descriptors(X, y) -> pd.DataFrame:
    """Structure of a stream in consecutive windows."""
    rows, previous = [], {}
    for start in range(0, len(X) - WINDOW + 1, WINDOW):
        xs, ys = X[start : start + WINDOW], y[start : start + WINDOW]
        centroids, spreads = {}, []
        for label in np.unique(ys[ys >= 0]):
            members = xs[ys == label]
            if len(members) >= 5:
                centroids[label] = members.mean(axis=0)
                spreads.append(
                    np.sqrt(((members - centroids[label]) ** 2).sum(axis=1).mean())
                )
        shared = [label for label in centroids if label in previous]
        points = np.array(list(centroids.values()))
        gaps = (
            np.linalg.norm(points[:, None] - points[None, :], axis=2)[
                np.triu_indices(len(points), 1)
            ]
            if len(points) > 1
            else np.array([])
        )
        rows.append(
            {
                "window": start // WINDOW,
                "clusters": len(centroids),
                "noise": float(np.mean(ys == -1)),
                "spread": float(np.mean(spreads)) if spreads else np.nan,
                "displacement": (
                    float(
                        np.mean(
                            [np.linalg.norm(centroids[l] - previous[l]) for l in shared]
                        )
                    )
                    if shared
                    else np.nan
                ),
                "separation": float(gaps.min()) if len(gaps) else np.nan,
            }
        )
        previous = centroids
    return pd.DataFrame(rows)


def compare(table: pd.DataFrame, value_columns, group_columns) -> pd.DataFrame:
    """Port against R on every value column, within every group; p-values corrected together."""
    rows = []
    for group, frame in table.groupby(group_columns, sort=False):
        group = group if isinstance(group, tuple) else (group,)
        for column in value_columns:
            reference = frame.loc[frame.source == "R", column].dropna().to_numpy()
            port = frame.loc[frame.source == "port", column].dropna().to_numpy()
            if len(reference) == 0 or len(port) == 0:
                continue
            rows.append(
                {
                    **dict(zip(group_columns, group)),
                    "quantity": column,
                    "reference_median": float(np.median(reference)),
                    "port_median": float(np.median(port)),
                    "p": safe_mannwhitney(port, reference),
                    "delta": cliffs_delta(port, reference),
                }
            )
    out = pd.DataFrame(rows)
    out["p_adj"] = multipletests(out["p"], alpha=ALPHA, method="holm")[1]
    out["different"] = (out["p_adj"] < ALPHA) & (
        out["delta"].abs() >= CLIFFS_DELTA_THRESHOLD
    )
    return out


def distribution_tests() -> pd.DataFrame:
    """Two-sample tests on the pooled samples of the generators with independent samples."""
    rows = []
    for dataset in INDEPENDENT_SAMPLES:
        pooled = {"R": [], "port": []}
        for source, _, X, y in sources(dataset):
            pooled[source].append(np.column_stack([X, y]))
        reference, port = (np.vstack(pooled[s]) for s in ("R", "port"))
        labels = np.unique(reference[:, -1])
        counts = [
            [np.sum(data[:, -1] == label) for label in labels]
            for data in (reference, port)
        ]
        rows.append(
            {
                "dataset": dataset,
                "test": "cluster proportions",
                "statistic": np.nan,
                "p": float(chi2_contingency(counts).pvalue),
                "samples": len(reference),
            }
        )
        for label in labels:
            for axis, name in enumerate("xy"):
                test = ks_2samp(
                    reference[reference[:, -1] == label, axis],
                    port[port[:, -1] == label, axis],
                )
                rows.append(
                    {
                        "dataset": dataset,
                        "test": f"{name} of label {int(label)}",
                        "statistic": float(test.statistic),
                        "p": float(test.pvalue),
                        "samples": len(reference),
                    }
                )
    out = pd.DataFrame(rows)
    out["p_adj"] = multipletests(out["p"], alpha=ALPHA, method="holm")[1]
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    windows, runs, noise, behaviour = [], [], [], []
    for dataset in DATASETS:
        for source, run, X, y in sources(dataset):
            described = window_descriptors(X, y)
            windows.append(described.assign(dataset=dataset, source=source, run=run))
            runs.append(
                {
                    "dataset": dataset,
                    "source": source,
                    "run": run,
                    **described[DESCRIPTORS].mean(),
                }
            )
            if dataset == "random_rbf_events":
                for block in range(len(y) // BLOCK):
                    share = float(np.mean(y[block * BLOCK : (block + 1) * BLOCK] == -1))
                    noise.append(
                        {"source": source, "run": run, "block": block, "noise": share}
                    )
            if source == "port, own labels":
                continue
            for name, build in MODELS.items():
                results = prequential(build(STARTING_CLUSTERS[dataset]), X, y)
                behaviour.append(
                    {
                        "dataset": dataset,
                        "source": source,
                        "run": run,
                        "model": name,
                        **ari_summary(results),
                    }
                )
        print(f"{dataset} done", flush=True)

    pd.concat(windows).to_csv(OUT_DIR / "descriptors_by_window.csv", index=False)
    runs = pd.DataFrame(runs)
    runs.to_csv(OUT_DIR / "descriptors_by_run.csv", index=False)
    compare(runs, DESCRIPTORS, ["dataset"]).to_csv(
        OUT_DIR / "descriptor_tests.csv", index=False
    )
    pd.DataFrame(noise).to_csv(OUT_DIR / "rbf_noise_by_block.csv", index=False)
    behaviour = pd.DataFrame(behaviour)
    behaviour.to_csv(OUT_DIR / "behaviour_by_run.csv", index=False)
    compare(behaviour, ["ari_overall", "ari_steadystate"], ["dataset", "model"]).to_csv(
        OUT_DIR / "behaviour_tests.csv", index=False
    )
    distribution_tests().to_csv(OUT_DIR / "distribution_tests.csv", index=False)


if __name__ == "__main__":
    main()
