"""Confirmation of the improvement on the evaluation runs of the real-data experiment.

The final model of the improvement rounds, run by `run_confirmation.sh` on runs 0-31 of the
eight real datasets, joins the benchmark models and MicroTEDAClus+ of the real-data
experiment, whose tasks used the same streams and the same seeds of the search. Two analyses:

- the comparison of the final model with each other model in each dataset, as in the
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
from benchmark.real_artifacts import DATASETS, RUNS, complete_by_dataset
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
    real_tasks = real_tasks[real_tasks.run.isin(RUNS)]
    real = analysis.run_level(analysis.REAL)
    real = real[real.run.isin(RUNS)]
    parts, tasks = [real], [real_tasks]
    for name in CHAIN[1:]:
        out = results_dir / name
        if (out / "tasks.csv").exists():
            parts.append(analysis.run_level(out))
            tasks.append(pd.read_csv(out / "tasks.csv"))
    return pd.concat(parts), pd.concat(tasks)


def focus_table(results, focus, others, metric):
    """Medians of every model; B or W where the model in focus is better or worse than it."""
    compared = analysis.compared(results, metric, focus, others, DATASETS)
    lines = [f"{metric:22}" + "".join(f"{SHORT[d]:>10}" for d in DATASETS)]
    for model in (focus, *others):
        of_model = compared[compared.model == model].set_index("dataset")
        cells = []
        for dataset in DATASETS:
            if dataset not in of_model.index:
                cells.append(f"{'-':>10}")
                continue
            row = of_model.loc[dataset]
            mark = {"better": "B", "worse": "W"}.get(row.verdict, " ")
            cells.append(f"{row['median']:.3f}{mark}".rjust(10))
        lines.append(f"{model[:22]:22}" + "".join(cells))
    return "\n".join(lines)


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

    focus = FINAL if FINAL in models else "MicroTEDAClusPlus"
    others = [m for m in models if m != focus]
    for metric in analysis.QUALITY_METRICS:
        print("\n" + focus_table(complete, focus, others, metric))
    print("\nchain (paired, runs shared):")
    print(chain(results))


if __name__ == "__main__":
    main()
