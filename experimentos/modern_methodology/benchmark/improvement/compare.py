"""Comparison of a candidate with its control, on the runs that both finished.

Per dataset and metric, the paired test of the thesis (Wilcoxon signed-rank with
Holm's correction over the datasets of a metric, and a threshold on Cliff's
delta). The outcome is decided on the Overall ARI in every dataset and on the
Steady-State ARI in the datasets whose penultimate batch does not hold a single
class:

    +  significantly better somewhere and worse nowhere
    -  worse somewhere and better nowhere
    ±  both
    =  neither

The ARI at the fifth batch and the summary size are reported and do not decide.

Usage (from experimentos/modern_methodology):
    python -m benchmark.improvement.compare CANDIDATE [CONTROL]
"""

import sys

import pandas as pd

from benchmark import analysis
from benchmark.improvement.candidates import CANDIDATES
from benchmark.improvement.run import DATASETS, OUT
from pfc_shared.statistics import paired_comparisons

# their penultimate batch holds a single class in every run
NO_STEADY_STATE = {"insects_abrupt", "insects_incremental_reoccurring"}
DECIDING = {
    "ari_overall": DATASETS,
    "ari_steadystate": [d for d in DATASETS if d not in NO_STEADY_STATE],
}
REPORTED = {"ari_fifth": DATASETS, "mc_max": DATASETS, "mc_steadystate": DATASETS}
SHORT = {
    "insects_abrupt": "INS-A",
    "insects_incremental": "INS-I",
    "insects_incremental_reoccurring": "INS-R",
    "gas_drift": "GAS",
    "pendigits": "PEN",
    "shuttle": "SHU",
}
MARK = {"better": "+", "worse": "-", "same": " "}


def results(name: str) -> pd.DataFrame:
    return analysis.run_level(OUT / name)


def failures(name: str) -> pd.Series:
    tasks = pd.read_csv(OUT / name / "tasks.csv")
    return tasks[tasks.status != "OK"].groupby("dataset").size()


def comparison(candidate: str, control: str) -> pd.DataFrame:
    """One row per metric and dataset: medians, the test and its verdict."""
    a, b = results(control), results(candidate)
    rows = []
    for metric, datasets in {**DECIDING, **REPORTED}.items():
        pairs = {}
        for dataset in datasets:
            x = a[a.dataset == dataset].set_index("run")[metric]
            y = b[b.dataset == dataset].set_index("run")[metric]
            shared = x.index.intersection(y.index)
            if len(shared) > 1:
                pairs[dataset] = (x.loc[shared].to_numpy(), y.loc[shared].to_numpy())
        tested = paired_comparisons(pairs, metric)
        for dataset, row in tested.iterrows():
            rows.append(
                {
                    "metric": metric,
                    "dataset": dataset,
                    "deciding": metric in DECIDING,
                    **row.to_dict(),
                }
            )
    return pd.DataFrame(rows)


def outcome(table: pd.DataFrame) -> tuple[str, int, int]:
    deciding = table[table.deciding]
    gains = int((deciding.verdict == "better").sum())
    losses = int((deciding.verdict == "worse").sum())
    if gains and losses:
        return "±", gains, losses
    if gains:
        return "+", gains, losses
    if losses:
        return "-", gains, losses
    return "=", gains, losses


def report(candidate: str, control: str | None = None) -> str:
    control = control or CANDIDATES[candidate].control
    table = comparison(candidate, control)
    result, gains, losses = outcome(table)
    lines = [
        f"{candidate} against {control}: {result} ({gains} gains, {losses} losses)",
        "metric           " + "".join(f"{SHORT[d]:>18}" for d in DATASETS),
    ]
    for metric in [*DECIDING, *REPORTED]:
        of_metric = table[table.metric == metric].set_index("dataset")
        cells = []
        for dataset in DATASETS:
            if dataset not in of_metric.index:
                cells.append(f"{'':>18}")
                continue
            r = of_metric.loc[dataset]
            fmt = "{:.0f}" if metric.startswith("mc") else "{:.3f}"
            text = f"{fmt.format(r.control_median)}>{fmt.format(r['median'])}{MARK[r.verdict]}"
            cells.append(f"{text:>18}")
        lines.append(f"{metric:17}" + "".join(cells))
    runs = table[table.metric == "ari_overall"].set_index("dataset")["n"]
    lines.append(
        "shared runs      " + "".join(f"{int(runs.get(d, 0)):>18}" for d in DATASETS)
    )
    for name in (control, candidate):
        failed = failures(name)
        if len(failed):
            lines.append(f"failed tasks of {name}: {failed.to_dict()}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(report(*sys.argv[1:]))
