"""Tables and figures of the experiment.

Written under monografia/Modernization/artifacts/experiment/ from the results
of `benchmark.runner`. Every table and figure uses the runs that all of its
models finished on every dataset.

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
from scipy.special import logit

from benchmark import analysis
from benchmark.datasets import SCALABILITY_DIMENSIONS
from benchmark.display import (
    MAIN_DATASETS,
    MODEL_COLORS,
    MODEL_NAMES,
    SCALABILITY_DATASETS,
    SHORT_NAMES,
    columns,
    model_label,
)
from benchmark.evaluation import BATCH_SIZE
from benchmark.hyperparameters import Category
from benchmark.models import MODELS
from benchmark.runner import batches_file
from pfc_shared.latex import marked, panel_table, write_table
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
    "ari_fifth": "ARI at the fifth batch",
    "ari_diff": "Maximum ARI drop",
    "mc_max": "Largest summary",
    "mc_steadystate": "Steady-state summary",
    "time_fit_cum_ms": "Learning time of the evaluation segment, in milliseconds",
    "time_predict_cum_ms": "Prediction time of the evaluation segment, in milliseconds",
    "memory_retained_bytes": "Memory held at the end of the stream, in KiB",
    "memory_peak_bytes": "Peak memory during the evaluation, in KiB",
}
GROUPS = (
    "A value in bold is statistically indistinguishable from the best model of its dataset, "
    "and a value in red from the worst."
)


def _in_ms(value: str) -> str:
    return f"{float(value):.0f}"


def _in_kib(value: str) -> str:
    return f"{float(value) / 1024:.0f}"


def tier_panels(results, metrics, models, datasets, shown=str):
    """A panel per metric; `shown` rewrites the text of a median, such as to change its unit."""
    panels = []
    for metric in metrics:
        of_metric = analysis.tiers(results, metric, models, datasets)
        rows = []
        for model in models:
            of_model = of_metric[of_metric.model == model].set_index("dataset")
            values = {
                dataset: marked(shown(row["median"]), row["verdict"])
                for dataset, row in of_model.iterrows()
            }
            rows.append((model_label(model), values, str))
        panels.append((TITLES[metric], rows))
    return panels


def quality_tables(results, models, runs):
    panel_table(
        ARTIFACT_DIR / "quality.tex",
        f"Quality of the models: medians over {runs} runs. {GROUPS}",
        "tab:mod_quality",
        columns(MAIN_DATASETS),
        tier_panels(results, analysis.QUALITY_METRICS, models, MAIN_DATASETS),
    )
    panel_table(
        ARTIFACT_DIR / "scalability.tex",
        f"Quality of the models along the dimension of the data: medians over {runs} runs. {GROUPS}",
        "tab:mod_scalability",
        columns(SCALABILITY_DATASETS),
        tier_panels(
            results, ("ari_overall", "ari_steadystate"), models, SCALABILITY_DATASETS
        ),
    )


def summary_table(results, models, runs):
    """Size of the summaries, in the six datasets and along the dimension of the data."""
    for name, label, datasets, scope in (
        ("summary_size", "tab:mod_summary_size", MAIN_DATASETS, ""),
        (
            "summary_size_scalability",
            "tab:mod_summary_size_scalability",
            SCALABILITY_DATASETS,
            " along the dimension of the data",
        ),
    ):
        panel_table(
            ARTIFACT_DIR / f"{name}.tex",
            f"Number of structures in the summary that each model keeps of the stream{scope}: "
            f"medians over {runs} runs. {GROUPS}",
            label,
            columns(datasets),
            tier_panels(results, analysis.SUMMARY_METRICS, models, datasets),
        )


def cost_tables(models):
    """Time and memory of the models, from the passes that measure them."""
    if (
        not (analysis.TIMING / "batches").exists()
        or not (analysis.MEMORY / "tasks.csv").exists()
    ):
        return
    timing = analysis.timing_level()
    memory = analysis.memory_level()
    runs = analysis.complete_runs(timing, models)
    if analysis.complete_runs(memory, models) != runs:
        raise ValueError("The time and memory passes completed different runs")
    timing = analysis.on_complete_runs(timing, models)
    memory = analysis.on_complete_runs(memory, models)
    for name, label, datasets, scope in (
        ("cost", "tab:mod_cost", MAIN_DATASETS, ""),
        (
            "cost_scalability",
            "tab:mod_cost_scalability",
            SCALABILITY_DATASETS,
            " along the dimension of the data",
        ),
    ):
        panel_table(
            ARTIFACT_DIR / f"{name}.tex",
            f"Execution time and memory of the models{scope}, each task measured alone and its "
            f"memory counted from every allocation of its process: medians over {len(runs)} runs. "
            f"{GROUPS}",
            label,
            columns(datasets),
            tier_panels(timing, analysis.TIME_METRICS, models, datasets, shown=_in_ms)
            + tier_panels(
                memory, analysis.MEMORY_METRICS, models, datasets, shown=_in_kib
            ),
        )


def protocol_table(rules, previous):
    """MicroTEDAClus+ under the two protocols of hyperparameter selection."""
    shared = sorted(set(rules.run) & set(previous.run))
    rules, previous = rules[rules.run.isin(shared)], previous[previous.run.isin(shared)]
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
        f"protocol by rules, on the same streams: medians over {len(shared)} runs. A value in bold is "
        "significantly better than that of the previous protocol, and a value in red is significantly worse.",
        "tab:mod_protocols",
        columns(SHORT_NAMES),
        panels,
        first_column="Selection",
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
    )


def at_the_ends(results, ranges, tolerance=0.05) -> pd.DataFrame:
    """Share of the runs whose selected value lies at an end of its search range.

    A value is at an end if it is within `tolerance` of the width of the range
    from it, in the scale of the search.
    """
    rows = []
    of_models = ranges[ranges.model.isin(set(results.model))]
    for (model, name), of_name in of_models.groupby(["model", "hyperparameter"]):
        column = f"param:{name}"
        selected = results.loc[results.model == model, ["dataset", "run", column]]
        merged = of_name.merge(selected, on=["dataset", "run"]).dropna(subset=[column])
        if merged.empty:
            continue
        if "logit" in merged and merged["logit"].iloc[0]:
            scale = logit
        elif merged["log"].iloc[0]:
            scale = np.log
        else:
            scale = lambda values: values  # noqa: E731
        low, high, value = (
            scale(merged[c].astype(float)) for c in ("low", "high", column)
        )
        width = (high - low).where(high > low)
        position = (value - low) / width
        merged = merged.assign(
            lower=position <= tolerance, upper=position >= 1 - tolerance
        )
        shares = merged[width.notna()].groupby("dataset")[["lower", "upper"]].mean()
        for dataset, row in shares.iterrows():
            rows.append(
                {
                    "model": model,
                    "hyperparameter": name,
                    "dataset": dataset,
                    "lower": row["lower"],
                    "upper": row["upper"],
                }
            )
    return pd.DataFrame(rows)


def ends_table(ends, models, runs):
    percent = "{:.0f}".format
    panels = []
    for end, title in (("lower", "At the lower end"), ("upper", "At the upper end")):
        rows = []
        for model in models:
            for name in MODELS[model].HYPERPARAMETERS:
                of_name = ends[(ends.model == model) & (ends.hyperparameter == name)]
                values = (100 * of_name.set_index("dataset")[end]).to_dict()
                escaped = name.replace("_", r"\_")
                rows.append(
                    (rf"{model_label(model)}, \texttt{{{escaped}}}", values, percent)
                )
        panels.append((title, rows))
    panel_table(
        ARTIFACT_DIR / "ends.tex",
        f"Percentage of the {runs} runs in which the selected value of a hyperparameter lies within a "
        "twentieth of the width of its search range from one of its ends, in the scale of the search.",
        "tab:mod_ends",
        columns(SHORT_NAMES),
        panels,
        first_column="Model and hyperparameter",
    )


def _bands(ax, x, values, model):
    """Median and interquartile band of the columns of `values`."""
    q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75], axis=0)
    ax.plot(
        x, median, color=MODEL_COLORS[model], linewidth=1.5, label=MODEL_NAMES[model]
    )
    ax.fill_between(x, q1, q3, color=MODEL_COLORS[model], alpha=0.15)


def _save(fig, axes, name, columns_of_legend=4, directory=ARTIFACT_DIR):
    handles, labels = np.ravel(axes)[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=columns_of_legend,
        bbox_to_anchor=(0.5, 1.12),
    )
    plt.tight_layout()
    out = directory / "plots" / name
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


def dimension_figure(results, models):
    """ARI against the dimension of the data, early in the stream and at its end."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, metric in zip(axes, ("ari_fifth", "ari_steadystate")):
        for model in models:
            of_model = results[results.model == model]
            values = np.column_stack(
                [
                    of_model[of_model.dataset == dataset].sort_values("run")[metric]
                    for dataset in SCALABILITY_DATASETS
                ]
            )
            _bands(ax, SCALABILITY_DIMENSIONS, values, model)
        ax.set_xscale("log", base=2)
        ax.set_xticks(SCALABILITY_DIMENSIONS, [str(d) for d in SCALABILITY_DIMENSIONS])
        ax.set_xlabel("Dimension")
        ax.set_title(TITLES[metric], fontsize=10)
    axes[0].set_ylabel("ARI")
    _save(fig, axes, "dimension_scalability.png")


def stream_figure(out, models, runs):
    """ARI along the stream in the lowest and the highest dimensions of the data."""
    datasets = ("SCL-2", "SCL-64")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, dataset in zip(axes, datasets):
        for model in models:
            values = np.vstack(
                [
                    pd.read_csv(batches_file(out, dataset, model, run))[
                        "ari"
                    ].to_numpy()
                    for run in runs
                ]
            )
            _bands(ax, (np.arange(values.shape[1]) + 1) * BATCH_SIZE, values, model)
        ax.set_xlabel("Samples of the evaluation segment")
        ax.set_title(f"Dimension {dataset.split('-')[1]}", fontsize=10)
    axes[0].set_ylabel("ARI")
    _save(fig, axes, "stream_scalability.png")


def main():
    results = analysis.run_level(analysis.MAIN)
    models = sys.argv[1:] or [m for m in MODELS if m in set(results.model)]
    runs = analysis.complete_runs(results, models)
    complete = analysis.on_complete_runs(results, models)
    print(f"{len(models)} models, {len(runs)} runs complete for all of them")

    quality_tables(complete, models, len(runs))
    summary_table(complete, models, len(runs))
    hyperparameter_table(models)
    ranges = analysis.search_ranges(analysis.MAIN, tuple(runs))
    range_table(ranges, len(runs))
    ends_table(at_the_ends(complete, ranges), models, len(runs))
    dimension_figure(complete, models)
    stream_figure(analysis.MAIN, models, runs)

    cost_tables(models)

    if (analysis.PREVIOUS_PROTOCOL / "tasks.csv").exists():
        previous = analysis.run_level(analysis.PREVIOUS_PROTOCOL)
        protocol_table(results[results.model == "MicroTEDAClusPlus"], previous)


if __name__ == "__main__":
    main()
