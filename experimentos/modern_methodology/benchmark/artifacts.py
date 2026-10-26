"""Tables and figures of the experiment, and the parts that those of the real data share.

Written under monografia/Modernization/artifacts/experiment/ from the results of
`run_modern_pilot.sh` and `run_modern_main.sh`. A task fails when its model exceeds
the limit of its summary or of its memory. A model that failed in more than half of
the runs of a dataset is left out of that dataset, and the other models are
compared over the runs that all of them finished on it, so a failed task removes
its run from its dataset only.

Usage (from experimentos/modern_methodology):
    python -m benchmark.artifacts [MODEL ...]
The models default to those of the experiment.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from benchmark import analysis, range_check
from benchmark.datasets import SCALABILITY_DIMENSIONS
from benchmark.display import (
    MAIN_DATASETS,
    MODEL_NAMES,
    SCALABILITY_DATASETS,
    SHORT_NAMES,
    columns,
    model_label,
)
from benchmark.evaluation import BATCH_SIZE
from benchmark.hyperparameters import Category
from benchmark.models import MODELS
from benchmark.runner import RUNS, batches_file
from pfc_shared.figures import band
from pfc_shared.latex import focus_rows, marked, panel_table, write_table
from pfc_shared.statistics import METRICS

ARTIFACT_DIR = (
    analysis.ROOT.parent.parent
    / "monografia"
    / "Modernization"
    / "artifacts"
    / "experiment"
)
TITLES = {
    "ari_overall": "Overall ARI",
    "ari_steadystate": "Steady-State ARI",
    "mc_max": "Largest summary",
    "mc_steadystate": "Steady-state summary",
    "time_fit_cum_ms": "Learning time [ms]",
    "time_predict_cum_ms": "Prediction time [ms]",
    "memory_retained_bytes": "Memory held at the end of the stream [KiB]",
    "memory_peak_bytes": "Peak memory [KiB]",
}
SUBJECT = r"\modelname{MicroTEDAClus+} and the benchmark models"
DATASETS = list(SHORT_NAMES)


def _in_ms(value: float) -> str:
    return f"{value:.0f}"


def _in_kib(value: float) -> str:
    return f"{value / 1024:.0f}"


def focus_panels(results, metrics, models, datasets, shown=None):
    """A panel per metric: the first of `models`, in focus, against each of the others.

    `shown` gives the text of a median, such as in another unit; by default, the
    format of its metric.
    """
    focus, *others = models
    panels = []
    for metric in metrics:
        compared = analysis.compared(results, metric, focus, others, datasets)
        text = shown or METRICS[metric]["fmt"].format
        panels.append(
            (TITLES[metric], focus_rows(compared, focus, others, model_label, text))
        )
    return panels


def failures(tasks: pd.DataFrame) -> pd.DataFrame:
    """Number of failed tasks of every model in every dataset."""
    failed = tasks.assign(failed=tasks.status != "OK")
    return failed.pivot_table(
        index="model", columns="dataset", values="failed", aggfunc="sum"
    )


def complete_by_dataset(results, tasks, models, datasets):
    """The results of each dataset over the runs that its models finished, the number of
    those runs in each dataset where the first of `models`, in focus, stays, and the
    datasets that leave it out.

    A model that failed in more than half of the runs of a dataset is left out of it.
    """
    kept, runs, without_focus = [], {}, []
    for dataset in datasets:
        attempted = tasks[tasks.dataset == dataset]
        finished = attempted[attempted.status == "OK"].groupby("model").size()
        staying = [
            m for m in models if finished.get(m, 0) > attempted.run.nunique() / 2
        ]
        of_dataset = analysis.on_complete_runs(results, staying, [dataset])
        kept.append(of_dataset)
        if models[0] in staying:
            runs[dataset] = of_dataset.run.nunique()
        else:
            without_focus.append(dataset)
    return pd.concat(kept), runs, without_focus


def failure_table(tasks, models, datasets, path, label, caption, short):
    """Failed runs of every model in every dataset, each a summary beyond the limit."""
    counts = failures(tasks)
    kinds = set(
        tasks.status[tasks.status != "OK"].str.extract(r"^ERROR: (\w+)", expand=False)
    )
    assert kinds <= {
        "SummaryLimitExceeded"
    }, f"the caption names only the summary limit: {kinds}"
    rows = [
        (
            model_label(model),
            counts.loc[model].to_dict() if model in counts.index else {},
            "{:.0f}".format,
        )
        for model in models
    ]
    panel_table(
        path,
        caption,
        label,
        columns(datasets),
        [("Failed runs", rows)],
        short=short,
    )


def run_text(runs: dict) -> str:
    """The number of runs compared, with the datasets that have fewer."""
    names = columns(runs)
    most = max(runs.values())
    fewer = [f"{n} in {names[d]}" for d, n in runs.items() if n < most]
    return f"{most} runs" + (f", except {', '.join(fewer)}" if fewer else "")


def left_out_text(complete, models, datasets, without_focus) -> str:
    """The sentence of a caption that explains its dashes; empty if there are none."""
    without = [d for d in datasets if d in without_focus]
    left_out = any(
        model not in set(complete[complete.dataset == dataset].model)
        for dataset in datasets
        for model in models
    )
    if not left_out:
        return ""
    return (
        " A dash marks a model that failed in more than half of the runs of a dataset"
        + (
            r", and every model of a dataset that leaves \MicroTEDAClusPlus{} out, "
            "where no comparison is made"
            if without
            else ""
        )
        + "."
    )


def cost_by_dataset(complete, models, datasets, without_focus, timing_dir, memory_dir):
    """Time and memory in each dataset over the runs that both passes completed for
    every model kept in it, and the number of those runs where the first of `models`
    stays."""
    timing = analysis.timing_level(timing_dir)
    memory = analysis.memory_level(memory_dir)
    kept_timing, kept_memory, runs = [], [], {}
    for dataset in datasets:
        staying = [
            m for m in models if m in set(complete[complete.dataset == dataset].model)
        ]
        t = analysis.on_complete_runs(timing, staying, [dataset])
        m = analysis.on_complete_runs(memory, staying, [dataset])
        shared = sorted(set(t.run) & set(m.run))
        kept_timing.append(t[t.run.isin(shared)])
        kept_memory.append(m[m.run.isin(shared)])
        if dataset not in without_focus:
            runs[dataset] = len(shared)
    return pd.concat(kept_timing), pd.concat(kept_memory), runs


def range_ends_table(pilot_dirs, models, path, label, caption, short):
    """The largest share of the runs of a dataset at each end of the ranges, in a pilot,
    each model from the last pilot directory that holds it; an end that is a natural
    limit in every dataset is in italics."""
    shares = range_check.combined_ends(pilot_dirs, models)
    percent = "{:.0f}".format
    lines = [r"Category & Lower end & Upper end \\", r"\midrule"]
    for category in Category:
        cells = []
        for end in ("lower", "upper"):
            of_end = shares[(shares.category == category.value) & (shares.end == end)]
            checked = of_end[~of_end.natural]
            if len(checked):
                cells.append(percent(100 * checked.share.max()))
            elif len(of_end):
                cells.append(rf"\textit{{{percent(100 * of_end.share.max())}}}")
            else:
                cells.append("-")
        lines.append(" & ".join([category.value.capitalize(), *cells]) + r" \\")
    write_table(path, caption, label, "l c c", lines, short)


def quality_tables(results, models, runs, without_focus):
    for name, label, datasets, scope in (
        ("quality", "tab:mod_quality", MAIN_DATASETS, ""),
        (
            "scalability",
            "tab:mod_scalability",
            SCALABILITY_DATASETS,
            " in the scalability streams",
        ),
    ):
        title = f"Clustering quality of {SUBJECT}{scope}"
        of_table = {d: n for d, n in runs.items() if d in datasets}
        panel_table(
            ARTIFACT_DIR / f"{name}.tex",
            f"{title}: medians over {run_text(of_table)}."
            + left_out_text(results, models, datasets, without_focus),
            label,
            columns(datasets),
            focus_panels(results, analysis.QUALITY_METRICS, models, datasets),
            short=title,
        )


def summary_table(results, models, runs, without_focus):
    """Size of the summaries in the six datasets; the scalability streams have a figure."""
    title = f"Summary size of {SUBJECT}"
    of_table = {d: n for d, n in runs.items() if d in MAIN_DATASETS}
    panel_table(
        ARTIFACT_DIR / "summary_size.tex",
        f"{title}: medians over {run_text(of_table)}."
        + left_out_text(results, models, MAIN_DATASETS, without_focus),
        "tab:mod_summary_size",
        columns(MAIN_DATASETS),
        focus_panels(results, analysis.SUMMARY_METRICS, models, MAIN_DATASETS),
        short=title,
    )


def cost_table(timing, memory, models, runs, note):
    """Time and memory in the six datasets; the scalability streams have a figure."""
    title = f"Execution time and memory of {SUBJECT}"
    panel_table(
        ARTIFACT_DIR / "cost.tex",
        f"{title}: medians over {run_text(runs)}." + note,
        "tab:mod_cost",
        columns(MAIN_DATASETS),
        focus_panels(timing, analysis.TIME_METRICS, models, MAIN_DATASETS, _in_ms)
        + focus_panels(memory, analysis.MEMORY_METRICS, models, MAIN_DATASETS, _in_kib),
        short=title,
    )


def protocol_table(rules, previous):
    """MicroTEDAClus+ under the two protocols of hyperparameter selection, in every
    dataset over the runs that both finished."""
    shared = {
        dataset: len(
            set(rules.run[rules.dataset == dataset])
            & set(previous.run[previous.dataset == dataset])
        )
        for dataset in DATASETS
    }
    panels = []
    for metric in ("ari_overall", "ari_steadystate"):
        comparison = analysis.against(previous, rules, metric)
        fmt = METRICS[metric]["fmt"].format
        by_rules = {
            dataset: marked(fmt(row["median"]), row["verdict"])
            for dataset, row in comparison.iterrows()
        }
        panels.append(
            (
                TITLES[metric],
                [
                    ("Previous protocol", comparison["control_median"], fmt),
                    ("Protocol by rules", by_rules, str),
                ],
            )
        )
    panel_table(
        ARTIFACT_DIR / "protocols.tex",
        r"\MicroTEDAClusPlus{} with its hyperparameters selected by the previous protocol and by the "
        f"protocol by rules, on the same streams: medians over "
        f"{run_text({d: n for d, n in shared.items() if n})}.",
        "tab:mod_protocols",
        columns(SHORT_NAMES),
        panels,
        first_column="Selection",
        short=r"\MicroTEDAClusPlus{} under the two protocols of hyperparameter selection",
    )


def hyperparameter_table(models):
    lines = [r"Model & Hyperparameter & Category \\"]
    for model in models:
        lines.append(r"\midrule")
        for position, (name, category) in enumerate(
            MODELS[model].HYPERPARAMETERS.items()
        ):
            label = model_label(model) if position == 0 else ""
            escaped = name.replace("_", r"\_")
            lines.append(
                rf"{label} & \texttt{{{escaped}}} & {category.value.capitalize()} \\"
            )
    write_table(
        ARTIFACT_DIR / "hyperparameters.tex",
        "Hyperparameters that the search selects for each model, and the category that gives each its range.",
        "tab:mod_hyperparameters",
        "l l l",
        lines,
        "Hyperparameters of the models and their categories",
    )


def _number(value: float) -> str:
    """A positive number with two significant digits, as LaTeX."""
    if value >= 100:
        return f"{value:.0f}"
    if value >= 0.001:
        return f"{value:.2g}"
    mantissa, exponent = f"{value:.1e}".split("e")
    return rf"${mantissa} \times 10^{{{int(exponent)}}}$"


def _interval(low, high, integer):
    if integer:
        return f"{low:.0f} to {high:.0f}"
    return f"{_number(low)} to {_number(high)}"


def range_table(ranges, runs):
    """Ranges that the rules give in each dataset, for the categories that depend on the tuning segment."""
    categories = {
        Category.DISTANCE: False,
        Category.RELATIVE_VARIANCE: False,
        Category.HORIZON: True,
        Category.SUMMARY_SIZE: True,
        Category.MINIMUM_POINTS: True,
        Category.CLUSTERS: True,
    }
    of_category = ranges.drop_duplicates(["dataset", "run", "category"])
    medians = of_category.groupby(["dataset", "category"])[["low", "high"]].median()
    lines = [
        "Dataset & "
        + " & ".join(category.value.capitalize() for category in categories)
        + r" \\",
        r"\midrule",
    ]
    for dataset, short in SHORT_NAMES.items():
        if dataset not in medians.index:
            continue
        cells = [
            _interval(*medians.loc[(dataset, category.value)], integer)
            for category, integer in categories.items()
        ]
        lines.append(" & ".join([short, *cells]) + r" \\")
    write_table(
        ARTIFACT_DIR / "ranges.tex",
        f"Search ranges that the rules give to each category of hyperparameter: medians over {runs} runs of "
        "the two ends of each range.",
        "tab:mod_ranges",
        "l " + "c " * len(categories),
        lines,
        "Search ranges that the rules give",
    )


def _bands(ax, x, values, model, focus):
    """Median of the columns of `values`, with its interquartile band for the model in focus."""
    q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75], axis=0)
    band(ax, x, median, q1, q3, model, MODEL_NAMES[model], model == focus)


def _bands_by_dimension(ax, frame, metric, model, focus, scale=1.0):
    """Median of a metric of a model in each scalability stream, over the runs of that
    stream, with its interquartile band for the model in focus; a gap where the model is
    left out."""
    quartiles = []
    for dataset in SCALABILITY_DATASETS:
        values = frame.loc[
            (frame.model == model) & (frame.dataset == dataset), metric
        ].to_numpy(float)
        quartiles.append(
            np.quantile(values / scale, [0.25, 0.5, 0.75])
            if len(values)
            else [np.nan] * 3
        )
    q1, median, q3 = np.array(quartiles).T
    band(
        ax,
        SCALABILITY_DIMENSIONS,
        median,
        q1,
        q3,
        model,
        MODEL_NAMES[model],
        model == focus,
    )


def _legend_entries(axes):
    """Handles and labels of every model drawn in any of the axes, each once, in order."""
    entries = {}
    for ax in np.ravel(axes):
        for handle, label in zip(*ax.get_legend_handles_labels()):
            entries.setdefault(label, handle)
    return list(entries.values()), list(entries)


def _save(fig, axes, name, columns_of_legend=4, directory=ARTIFACT_DIR):
    handles, labels = _legend_entries(axes)
    plt.tight_layout()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=columns_of_legend,
        bbox_to_anchor=(0.5, 1.0),
    )
    out = directory / "plots" / name
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


def quality_figure(results, out, models):
    """ARI along the stream in the lowest and the highest dimensions of the data,
    and the Overall and Steady-State ARI against the dimension."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, dataset in zip(axes[0], ("SCL-2", "SCL-64")):
        of_dataset = results[results.dataset == dataset]
        runs = sorted(of_dataset.run.unique())
        for model in [m for m in models if m in set(of_dataset.model)]:
            values = np.vstack(
                [
                    pd.read_csv(batches_file(out, dataset, model, run))[
                        "ari"
                    ].to_numpy()
                    for run in runs
                ]
            )
            _bands(
                ax,
                (np.arange(values.shape[1]) + 1) * BATCH_SIZE,
                values,
                model,
                models[0],
            )
        ax.set_xlabel("Samples of the evaluation segment")
        ax.set_ylabel("ARI")
        ax.set_title(f"Along the stream, {dataset.split('-')[1]} dimensions")
    for ax, metric in zip(axes[1], analysis.QUALITY_METRICS):
        for model in models:
            _bands_by_dimension(ax, results, metric, model, models[0])
        ax.set_xscale("log", base=2)
        ax.set_xticks(SCALABILITY_DIMENSIONS, [str(d) for d in SCALABILITY_DIMENSIONS])
        ax.set_xlabel("Dimension")
        ax.set_ylabel("ARI")
        ax.set_title(TITLES[metric])
    _save(fig, axes, "scalability_quality.png")


def cost_figure(timing, memory, results, models):
    """Time, memory and summary size against the dimension of the data."""
    panels = [
        (timing, "time_fit_cum_ms", 1.0),
        (timing, "time_predict_cum_ms", 1.0),
        (memory, "memory_retained_bytes", 1024.0),
        (memory, "memory_peak_bytes", 1024.0),
        (results, "mc_max", 1.0),
        (results, "mc_steadystate", 1.0),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    for ax, (frame, metric, scale) in zip(np.ravel(axes), panels):
        for model in models:
            _bands_by_dimension(ax, frame, metric, model, models[0], scale)
        ax.set_xscale("log", base=2)
        ax.set_xticks(SCALABILITY_DIMENSIONS, [str(d) for d in SCALABILITY_DIMENSIONS])
        ax.set_yscale("log")
        ax.set_xlabel("Dimension")
        ax.set_ylabel(TITLES[metric])
    _save(fig, axes, "cost_scalability.png")


def main():
    results = analysis.run_level(analysis.MAIN)
    results = results[results.run.between(*RUNS)]
    tasks = pd.read_csv(analysis.MAIN / "tasks.csv")
    tasks = tasks[tasks.run.between(*RUNS)]
    models = sys.argv[1:] or [m for m in MODELS if m in set(tasks.model)]
    tasks = tasks[tasks.model.isin(models)]
    complete, runs, without_focus = complete_by_dataset(
        results, tasks, models, DATASETS
    )
    print(f"{len(models)} models; complete runs by dataset: {runs}")
    print("datasets that leave the model in focus out:", without_focus)

    if (tasks.status != "OK").any():
        failure_table(
            tasks,
            models,
            DATASETS,
            ARTIFACT_DIR / "failures.tex",
            "tab:mod_failures",
            "Number of runs in which a model failed. Every failure is a summary that "
            "exceeded a quarter of the tuning segment.",
            "Failed runs of the modernized experiment",
        )
    quality_tables(complete, models, runs, without_focus)
    summary_table(complete, models, runs, without_focus)
    hyperparameter_table(models)
    ranges = analysis.search_ranges(
        analysis.MAIN,
        tuple(sorted(tasks.run.unique())),
        tuple(d for d in DATASETS if d in set(tasks.dataset)),
    )
    range_table(ranges, ranges.run.nunique())
    quality_figure(complete, analysis.MAIN, models)

    if (analysis.TIMING / "batches").exists() and (
        analysis.MEMORY / "tasks.csv"
    ).exists():
        timing, memory, cost_runs = cost_by_dataset(
            complete, models, DATASETS, without_focus, analysis.TIMING, analysis.MEMORY
        )
        cost_table(
            timing,
            memory,
            models,
            {d: n for d, n in cost_runs.items() if d in MAIN_DATASETS},
            left_out_text(complete, models, MAIN_DATASETS, without_focus),
        )
        cost_figure(timing, memory, complete, models)

    if (analysis.PREVIOUS_PROTOCOL / "tasks.csv").exists():
        previous = analysis.run_level(analysis.PREVIOUS_PROTOCOL)
        previous = previous[previous.run.between(*RUNS)]
        focus = models[0]
        rules = results[(results.model == focus) & ~results.dataset.isin(without_focus)]
        protocol_table(rules, previous)

    if (analysis.PILOT / "tasks.csv").exists():
        range_ends_table(
            [analysis.PILOT],
            models,
            ARTIFACT_DIR / "range_ends.tex",
            "tab:mod_range_ends",
            "Largest share of the runs of a dataset in which the value that the search "
            "selects for a hyperparameter lies at an end of its range, within a twentieth "
            "of its width, in the pilot: percentages. A value in italics is that of a "
            "natural limit.",
            "Selected values at the ends of their search ranges",
        )


if __name__ == "__main__":
    main()
