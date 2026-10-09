"""Quality along the stream of the configurations that the reachability probe singles out.

For each probed dataset, the configuration with the highest tuning objective and
the one with the highest Overall ARI of the final model's space learn the whole
stream from its first sample, in the prequential manner of the protocol. The
batch ARI, the number of predicted clusters and the number of microclusters are
summarized in fifths of the stream, the first of which is the segment that the
hyperparameter search sees. The result tells whether a configuration that the
objective ranks low is worse from the start or only slower to mature.

Usage (from experimentos/proposed_methodology), after the probes of the final space:
    python -m model_improvement.maturation
"""

import numpy as np
import pandas as pd

import protocol
from model_improvement.experiments import FINAL
from model_improvement.harness import batch_diagnostics
from model_improvement.probe import PROBE_DIR
from model_improvement.variants import resolve
from streams import load_stream

DATASETS = ["cassini", "chameleon_ds3", "chameleon_ds4"]
SEGMENTS = 5
OUT_FILE = PROBE_DIR / "maturation.csv"


def singled_out(dataset: str) -> dict:
    """Hyperparameters of the two configurations, chosen on the means over the probed runs."""
    probe = pd.read_csv(PROBE_DIR / f"final_{dataset}.csv")
    means = probe.groupby("configuration")[["tuning_objective", "ari_overall"]].mean()
    picks = {
        "highest objective": means["tuning_objective"].idxmax(),
        "highest Overall ARI": means["ari_overall"].idxmax(),
    }
    hyperparameters = probe.groupby("configuration")[
        ["variance_limit", "decay", "noise_m"]
    ].first()
    return {name: hyperparameters.loc[index].to_dict() for name, index in picks.items()}


def main():
    model_class, _ = resolve(FINAL)
    rows = []
    for dataset in DATASETS:
        runs = sorted(pd.read_csv(PROBE_DIR / f"final_{dataset}.csv")["run"].unique())
        for name, params in singled_out(dataset).items():
            params["decay"] = int(params["decay"])
            for run in runs:
                X, y = load_stream(dataset, run)
                results = pd.DataFrame(
                    protocol.evaluate(
                        model_class(**params), X, y, batch_diagnostics=batch_diagnostics
                    )
                )
                segments = [
                    results.iloc[rows_of_segment]
                    for rows_of_segment in np.array_split(
                        np.arange(len(results)), SEGMENTS
                    )
                ]
                row = {"dataset": dataset, "configuration": name, "run": run}
                for position, segment in enumerate(segments, start=1):
                    row[f"ari_{position}"] = segment["ari"].median()
                    row[f"clusters_{position}"] = segment["n_pred"].median()
                    row[f"true_clusters_{position}"] = segment["n_true"].median()
                    row[f"microclusters_{position}"] = segment[
                        "microcluster_count"
                    ].iloc[-1]
                rows.append(row)
        print(f"{dataset} done", flush=True)
    pd.DataFrame(rows).to_csv(OUT_FILE, index=False)


if __name__ == "__main__":
    main()
