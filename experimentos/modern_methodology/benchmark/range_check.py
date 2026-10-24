"""Whether the search ranges hold the hyperparameters that the search selects.

An end of a range binds when, in at least a quarter of the runs of a dataset, the
selected value lies within a twentieth of the width of the range from it, in the
scale of the search. A model is judged only on the datasets where it finished
more than half of the runs, as the comparison of the models keeps it. An end that
is a natural limit of its category is reported apart and never fails the check.

Usage (from experimentos/modern_methodology):
    python -m benchmark.range_check DIR [DIR ...]
Every DIR holds the tasks of an experiment, a tune-only pilot or a complete one.
A model is judged on the tasks of the last DIR that holds it, so a later pilot of
some models replaces their tasks of an earlier one. The check exits with status 1
if an end that is not a natural limit binds.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from benchmark import analysis
from benchmark.artifacts import at_the_ends
from benchmark.hyperparameters import Category, natural_limit

TOLERANCE = 0.05  # of the width of a range, in the scale of the search
BINDING_SHARE = 0.25  # of the runs of a dataset


def selected_hyperparameters(out: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The tasks of OUT, and one row per finished task with its selected hyperparameters."""
    tasks = pd.read_csv(out / "tasks.csv")
    rows = [
        {
            "dataset": task.dataset,
            "model": task.model,
            "run": task.run,
            **{
                f"param:{name}": value
                for name, value in json.loads(task.hyperparameters).items()
            },
        }
        for task in tasks[tasks.status == "OK"].itertuples()
    ]
    return tasks, pd.DataFrame(rows)


def judged(tasks: pd.DataFrame) -> set:
    """(model, dataset) of the models that finished more than half of the runs of a dataset."""
    attempted = tasks.groupby(["model", "dataset"]).run.nunique()
    finished = tasks[tasks.status == "OK"].groupby(["model", "dataset"]).run.nunique()
    return {key for key, n in finished.items() if n > attempted[key] / 2}


def ends(out: Path, models=None) -> pd.DataFrame:
    """Share of the runs at each end of every range of the models of OUT, in the datasets
    where they are judged, with whether the end is a natural limit."""
    tasks, selected = selected_hyperparameters(out)
    if models is not None:
        tasks = tasks[tasks.model.isin(models)]
        selected = selected[selected.model.isin(models)]
    runs = tuple(sorted(tasks.run.unique()))
    datasets = tuple(sorted(tasks.dataset.unique()))
    ranges = analysis.search_ranges(out, runs, datasets)
    ranges = ranges[ranges.model.isin(set(tasks.model))]
    shares = at_the_ends(selected, ranges, TOLERANCE)
    keep = judged(tasks)
    shares = shares[
        [(m, d) in keep for m, d in zip(shares.model, shares.dataset)]
    ].reset_index(drop=True)
    of_range = ranges.groupby(["model", "hyperparameter", "dataset"]).agg(
        category=("category", "first"), high=("high", "min")
    )
    rows = []
    for row in shares.itertuples():
        info = of_range.loc[(row.model, row.hyperparameter, row.dataset)]
        category = Category(info.category)
        for end in ("lower", "upper"):
            rows.append(
                {
                    "model": row.model,
                    "hyperparameter": row.hyperparameter,
                    "category": category.value,
                    "dataset": row.dataset,
                    "end": end,
                    "share": getattr(row, end),
                    "natural": natural_limit(category, end, info.high),
                }
            )
    return pd.DataFrame(rows)


def combined_ends(directories) -> pd.DataFrame:
    """The ends of every model, each from the last directory that holds it."""
    holder = {}
    for out in directories:
        for model in pd.read_csv(out / "tasks.csv").model.unique():
            holder[model] = out
    by_directory = {}
    for model, out in holder.items():
        by_directory.setdefault(out, []).append(model)
    return pd.concat([ends(out, models) for out, models in by_directory.items()])


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("directories", type=Path, nargs="+")
    arguments = parser.parse_args()

    shares = combined_ends(arguments.directories)
    largest = (
        shares.sort_values("share", ascending=False)
        .groupby(["model", "hyperparameter", "end"], sort=False)
        .head(1)
    )
    binding = largest[largest.share >= BINDING_SHARE]
    pd.set_option("display.width", 200)
    print(
        f"Ends at which the selected value lies in at least {BINDING_SHARE:.0%} of the runs "
        "of a dataset:"
    )
    columns = ["model", "hyperparameter", "end", "share", "dataset"]
    for natural, title in ((False, "binding"), (True, "natural limits, not checked")):
        of_kind = binding[binding.natural == natural]
        print(f"\n{title}:")
        print(of_kind[columns].to_string(index=False) if len(of_kind) else "  none")
    if (~binding.natural).any():
        sys.exit("ERROR: an end that is not a natural limit binds")
    print("\nEvery end that is not a natural limit holds the selected values.")


if __name__ == "__main__":
    main()
