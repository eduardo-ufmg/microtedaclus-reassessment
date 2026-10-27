"""Run-level metrics and statistical comparisons of the experiment.

The metrics and the comparisons come from the shared package: the ARI and
summary-size metrics of a run, the comparison of a model in focus with each
other model of a dataset, and the paired comparison of two alternatives over the
same runs.
"""

import json
from functools import lru_cache
from pathlib import Path

import pandas as pd

from benchmark.datasets import DATASETS
from benchmark.hyperparameters import RULES
from benchmark.models import MODELS
from benchmark.preprocessing import segments
from benchmark.runner import batches_file, stream_length
from pfc_shared.metrics import (
    add_time_columns,
    microcluster_metrics,
    quality_metrics,
    time_metrics,
)
from pfc_shared.statistics import compare_with_focus, paired_comparisons

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "results_main"
PREVIOUS_PROTOCOL = ROOT / "results_previous_protocol"
REAL_MAIN = ROOT / "results_real_main"
TIMING_REAL = ROOT / "results_timing_real_main"
MEMORY_REAL = ROOT / "results_memory_real_main"
TIMING = ROOT / "results_timing"
MEMORY = ROOT / "results_memory"

QUALITY_METRICS = ("ari_overall", "ari_steadystate")
SUMMARY_METRICS = ("mc_max", "mc_steadystate")
TIME_METRICS = ("time_fit_cum_ms", "time_predict_cum_ms")
MEMORY_METRICS = ("memory_retained_bytes", "memory_peak_bytes")


def configuration(out: Path) -> dict:
    return json.loads((out / "configuration.json").read_text())


def length_factor(out: Path) -> int:
    return configuration(out)["length_factor"]


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


def timing_level(out: Path = TIMING) -> pd.DataFrame:
    """One row per task of the timing pass: its time metrics."""
    rows = []
    for file in sorted((out / "batches").glob("*/*/*.csv")):
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


def memory_level(out: Path = MEMORY) -> pd.DataFrame:
    """One row per task of the memory pass: its memory metrics."""
    tasks = pd.read_csv(out / "tasks.csv")
    return tasks[tasks.status == "OK"].drop(columns="status")


def complete_runs(results: pd.DataFrame, models, datasets=DATASETS) -> list:
    """Runs that every given model finished on every dataset."""
    of_models = results[results.model.isin(models) & results.dataset.isin(datasets)]
    finished = of_models.groupby("run").size()
    return sorted(finished.index[finished == len(datasets) * len(models)])


def on_complete_runs(results: pd.DataFrame, models, datasets=DATASETS) -> pd.DataFrame:
    return results[
        results.model.isin(models)
        & results.dataset.isin(datasets)
        & results.run.isin(complete_runs(results, models, datasets))
    ]


def compared(results: pd.DataFrame, metric: str, focus: str, others, datasets):
    """The model in focus against each other model in every dataset, as in
    `pfc_shared.statistics.compare_with_focus`."""
    return compare_with_focus(
        results[results.dataset.isin(datasets)], metric, focus, list(others)
    )


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


def search_ranges(out: Path, runs: tuple, datasets=tuple(DATASETS)) -> pd.DataFrame:
    """Search range of every hyperparameter of the models of OUT, in every dataset and run.

    The ranges come from the streams again, by the rules of the protocol of
    OUT, which takes long for the longest streams, so they are kept in
    OUT/search_ranges.csv once computed.
    """
    stored = out / "search_ranges.csv"
    if stored.exists():
        ranges = pd.read_csv(stored)
        if set(runs) <= set(ranges.run):
            return ranges[ranges.run.isin(runs)]
    rules = RULES[configuration(out)["protocol"]]
    models = set(pd.read_csv(out / "tasks.csv").model)
    rows = []
    for dataset in datasets:
        for run in runs:
            (X_tuning, y_tuning), _ = segments(
                dataset, run, stream_length(dataset, length_factor(out))
            )
            of_category = rules(X_tuning, y_tuning)
            for model, model_class in MODELS.items():
                if model not in models:
                    continue
                for name, category in model_class.HYPERPARAMETERS.items():
                    rows.append(
                        {
                            "dataset": dataset,
                            "run": run,
                            "model": model,
                            "hyperparameter": name,
                            "category": category.value,
                            "low": of_category[category].low,
                            "high": of_category[category].high,
                            "log": of_category[category].log,
                            "logit": of_category[category].logit,
                        }
                    )
    ranges = pd.DataFrame(rows)
    ranges.to_csv(stored, index=False)
    return ranges
