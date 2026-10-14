"""Run-level metrics and statistical comparisons of the experiment.

The metrics and the comparisons are those of the previous experiment, from the
shared package: the ARI and summary-size metrics of a run, the comparison of
every model with the best and with the worst of a dataset, and the paired
comparison of two alternatives over the same runs.
"""

import json
from functools import lru_cache
from pathlib import Path

import pandas as pd

from benchmark.datasets import DATASETS, load_stream
from benchmark.hyperparameters import ranges_of_kinds
from benchmark.models import MODELS
from benchmark.runner import batches_file, stream_length
from benchmark.tuning import split
from pfc_shared.metrics import (
    add_time_columns,
    microcluster_metrics,
    quality_metrics,
    time_metrics,
)
from pfc_shared.statistics import METRICS, paired_comparisons, process_metric

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "results_main"
PREVIOUS_PROTOCOL = ROOT / "results_previous_protocol"
TIMING = ROOT / "results_timing"
MEMORY = ROOT / "results_memory"

QUALITY_METRICS = ("ari_overall", "ari_steadystate", "ari_fifth", "ari_diff")
SUMMARY_METRICS = ("mc_max", "mc_steadystate")
TIME_METRICS = ("time_fit_cum_ms", "time_predict_cum_ms")
MEMORY_METRICS = ("memory_retained_bytes", "memory_peak_bytes")


def length_factor(out: Path) -> int:
    return json.loads((out / "configuration.json").read_text())["length_factor"]


@lru_cache(maxsize=None)
def run_level(out: Path) -> pd.DataFrame:
    """One row per finished task: its metrics, its search and its hyperparameters."""
    tasks = pd.read_csv(out / "tasks.csv")
    rows = []
    for task in tasks[tasks.status == "OK"].itertuples():
        batches = pd.read_csv(batches_file(out, task.dataset, task.model, task.run))
        rows.append(
            {
                "dataset": task.dataset,
                "model": task.model,
                "run": task.run,
                **quality_metrics(batches),
                **microcluster_metrics(batches),
                "objective": task.objective,
                "failed_trials": task.failed_trials,
                "tuning_seconds": task.tuning_seconds,
                "evaluation_seconds": task.evaluation_seconds,
                **{
                    f"param:{name}": value
                    for name, value in json.loads(task.hyperparameters).items()
                },
            }
        )
    return pd.DataFrame(rows)


def timing_level() -> pd.DataFrame:
    """One row per task of the timing pass: its time metrics."""
    rows = []
    for file in sorted((TIMING / "batches").glob("*/*/*.csv")):
        dataset, model = file.parts[-3:-1]
        batches = add_time_columns(pd.read_csv(file))
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "run": int(file.stem),
                **time_metrics(batches),
            }
        )
    return pd.DataFrame(rows)


def memory_level() -> pd.DataFrame:
    """One row per task of the memory pass: its memory metrics."""
    tasks = pd.read_csv(MEMORY / "tasks.csv")
    return tasks[tasks.status == "OK"].drop(columns="status")


def complete_runs(results: pd.DataFrame, models) -> list:
    """Runs that every given model finished on every dataset."""
    of_models = results[results.model.isin(models)]
    finished = of_models.groupby("run").size()
    return sorted(finished.index[finished == len(DATASETS) * len(models)])


def on_complete_runs(results: pd.DataFrame, models) -> pd.DataFrame:
    return results[
        results.model.isin(models) & results.run.isin(complete_runs(results, models))
    ]


def tiers(results: pd.DataFrame, metric: str, models, datasets) -> pd.DataFrame:
    """Median of every model in every dataset, and whether it is with the best or the worst.

    Columns: dataset, model, median and verdict, which is "better" for a model
    in the group of the best only, "worse" in that of the worst only, and
    "same" otherwise.
    """
    cells = process_metric(
        results[results.dataset.isin(datasets)], metric, METRICS[metric], list(models)
    )
    rows = []
    for dataset, of_dataset in cells.iterrows():
        for model, cell in of_dataset.items():
            value, _, tags = cell.partition(" [")
            verdict = {"B]": "better", "W]": "worse"}.get(tags, "same")
            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "median": value.split(" (")[0],
                    "verdict": verdict,
                }
            )
    return pd.DataFrame(rows)


def against(control: pd.DataFrame, variant: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Paired comparison, in every dataset, of two alternatives over the runs they share."""
    pairs = {}
    for dataset in DATASETS:
        a = control[control.dataset == dataset].set_index("run")[metric]
        b = variant[variant.dataset == dataset].set_index("run")[metric]
        shared = a.index.intersection(b.index)
        if len(shared):
            pairs[dataset] = (a.loc[shared].to_numpy(), b.loc[shared].to_numpy())
    return paired_comparisons(pairs, metric).rename_axis("dataset")


def search_ranges(out: Path, runs: tuple) -> pd.DataFrame:
    """Search range of every hyperparameter of every model, in every dataset and run.

    The ranges come from the streams again, which takes long for the longest
    streams, so they are kept in OUT/search_ranges.csv once computed.
    """
    stored = out / "search_ranges.csv"
    if stored.exists():
        ranges = pd.read_csv(stored)
        if set(runs) <= set(ranges.run):
            return ranges[ranges.run.isin(runs)]
    rows = []
    for dataset in DATASETS:
        for run in runs:
            X, y = load_stream(dataset, run, stream_length(dataset, length_factor(out)))
            (X_tuning, y_tuning), _ = split(X, y)
            of_kind = ranges_of_kinds(X_tuning, y_tuning)
            for model, model_class in MODELS.items():
                for name, kind in model_class.HYPERPARAMETERS.items():
                    rows.append(
                        {
                            "dataset": dataset,
                            "run": run,
                            "model": model,
                            "hyperparameter": name,
                            "kind": kind.value,
                            "low": of_kind[kind].low,
                            "high": of_kind[kind].high,
                            "log": of_kind[kind].log,
                        }
                    )
    ranges = pd.DataFrame(rows)
    ranges.to_csv(stored, index=False)
    return ranges
