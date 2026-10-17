"""Confirmation of the improvement on the evaluation runs of the real-data experiment.

The final model of the improvement rounds, run by `run_confirmation.sh` on runs 0-31 of the
eight real datasets, joins the benchmark models and MicroTEDAClus+ of the real-data
experiment, whose tasks used the same streams and the same seeds of the search. Two analyses:

- the comparison of every model with the best and the worst of each dataset, as in the
  tables of the real-data experiment, with the same treatment of failed runs;
- the paired comparisons of the chain: each step against its predecessor, from
  MicroTEDAClus+ (the control, which the copied engine reproduces exactly) to the final model.

Usage (from experimentos/modern_methodology):
    python -m benchmark.improvement.confirmation [--results DIR]
"""

import argparse
from pathlib import Path

import pandas as pd

from benchmark import analysis
from benchmark.improvement.compare import DECIDING, MARK, NO_STEADY_STATE
from benchmark.real_artifacts import DATASETS, complete_by_dataset
from pfc_shared.statistics import paired_comparisons

FINAL = "no_noise_link"
CHAIN = ["MicroTEDAClusPlus", "low_variance", "no_noise", FINAL]
BENCHMARKS = ["DenStream", "DBSTREAM", "BIRCH", "MiniBatchKMeans", "BICO"]
SHORT = {
    "insects_abrupt": "INS-A",
    "insects_incremental": "INS-I",
    "insects_incremental_reoccurring": "INS-R",
    "gas_drift": "GAS",
    "kddcup99": "KDD",
    "covertype": "COV",
    "pendigits": "PEN",
    "shuttle": "SHU",
}
# their penultimate batch holds a single class in every run (checked on the real-data runs)
SINGLE_CLASS_END = NO_STEADY_STATE | {"covertype"}


def load(results_dir: Path):
    real_tasks = pd.read_csv(analysis.REAL / "tasks.csv")
    real = analysis.run_level(analysis.REAL)
    parts, tasks = [real], [real_tasks]
    for name in CHAIN[1:]:
        out = results_dir / name
        if (out / "tasks.csv").exists():
            parts.append(analysis.run_level(out))
            tasks.append(pd.read_csv(out / "tasks.csv"))
    return pd.concat(parts), pd.concat(tasks)


def tier_table(results, models, metric):
    tiers = analysis.tiers(results, metric, models, DATASETS)
    lines = [f"{metric:22}" + "".join(f"{SHORT[d]:>10}" for d in DATASETS)]
    for model in models:
        of_model = tiers[tiers.model == model].set_index("dataset")
        cells = []
        for dataset in DATASETS:
            if dataset not in of_model.index:
                cells.append(f"{'-':>10}")
                continue
            row = of_model.loc[dataset]
            mark = {"better": "B", "worse": "W", "same": " "}[row.verdict]
            cells.append(f"{row['median'] + mark:>10}")
        lines.append(f"{model[:22]:22}" + "".join(cells))
    return "\n".join(lines), tiers


def chain(results):
    lines = []
    for control, step in zip(CHAIN, CHAIN[1:]):
        if step not in set(results.model):
            continue
        for metric in DECIDING:
            pairs = {}
            for dataset in DATASETS:
                if metric == "ari_steadystate" and dataset in SINGLE_CLASS_END:
                    continue
                a = results[(results.model == control) & (results.dataset == dataset)]
                b = results[(results.model == step) & (results.dataset == dataset)]
                a, b = a.set_index("run")[metric], b.set_index("run")[metric]
                shared = a.index.intersection(b.index)
                if len(shared) > 1:
                    pairs[dataset] = (
                        a.loc[shared].to_numpy(),
                        b.loc[shared].to_numpy(),
                    )
            tested = paired_comparisons(pairs, metric)
            cells = "  ".join(
                f"{SHORT[d]} {r.control_median:.3f}>{r['median']:.3f}{MARK[r.verdict]}"
                for d, r in tested.iterrows()
            )
            lines.append(f"{step} vs {control}, {metric}: {cells}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--results", type=Path, default=analysis.ROOT / "results_confirmation"
    )
    arguments = parser.parse_args()

    results, tasks = load(arguments.results)
    models = (
        BENCHMARKS
        + ["MicroTEDAClusPlus"]
        + ([FINAL] if FINAL in set(results.model) else [])
    )
    complete, runs = complete_by_dataset(results, tasks, models)
    print("complete runs by dataset:", {SHORT[d]: n for d, n in runs.items()})
    failed = tasks[(tasks.status != "OK") & tasks.model.isin(models)]
    print("failed tasks:", failed.groupby(["model", "dataset"]).size().to_dict())

    best_counts = {}
    for metric in ("ari_overall", "ari_steadystate", "ari_fifth", "ari_diff"):
        text, tiers = tier_table(complete, models, metric)
        print("\n" + text)
        if FINAL in models:
            of_final = tiers[tiers.model == FINAL]
            if metric == "ari_steadystate":
                of_final = of_final[~of_final.dataset.isin(SINGLE_CLASS_END)]
            best_counts[metric] = (
                int((of_final.verdict == "better").sum()),
                int((of_final.verdict == "worse").sum()),
                len(of_final),
            )
    if best_counts:
        print(
            "\nfinal model: datasets with the best (B) / with the worst (W) / compared"
        )
        for metric, (b, w, n) in best_counts.items():
            print(f"  {metric}: {b} / {w} / {n}")
    print("\nchain (paired, runs shared):")
    print(chain(results))


if __name__ == "__main__":
    main()
