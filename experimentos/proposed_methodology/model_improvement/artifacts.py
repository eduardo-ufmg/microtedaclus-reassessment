"""LaTeX tables and figures of the Model Improvement chapter.

Everything is written under monografia/ModelImprovement/artifacts/, one folder
per experiment, from the result files of the runner, of the serial timing pass
and of the reachability probe. In every table a value in bold is significantly
better than the value of its control and a value in red is significantly worse
(see analysis.py for the test).

Usage (from experimentos/proposed_methodology):
    python -m model_improvement.artifacts            # everything that has results
    python -m model_improvement.artifacts --summary  # print the decision of every comparison
"""

import argparse
from functools import lru_cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, ScalarFormatter
import numpy as np
import pandas as pd

import protocol
from compute_metrics import add_time_columns, is_valid_result, scalar_metrics
from editorial_config import dataset_display_name
from model_improvement import analysis
from model_improvement.experiments import (
    CORRECTED,
    DATASETS,
    DEVELOPMENT_RUNS,
    EXPERIMENT_BY_KEY,
    EXPERIMENTS,
    FINAL,
    IMPROVED,
    STAGE_NAMES,
    VALIDATION_RUNS,
)
from model_improvement.harness import RESULTS_DIR, result_paths
from model_improvement.variants import VARIANTS
from pfc_shared import latex
from pfc_shared.latex import marked, write_table
from statistic_analysis import METRICS
from streams import load_stream

ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = ROOT.parent.parent / "monografia" / "ModelImprovement" / "artifacts"
TIMING_FILE = RESULTS_DIR / "_timing" / "timing.csv"
PROBE_DIR = RESULTS_DIR / "_probe"

SHORT_NAMES = {
    "benchmark_1": "B1",
    "benchmark_2": "B2",
    "random_rbf_events": "RBF",
    "cassini": "CSN",
    "chameleon_ds3": "DS3",
    "chameleon_ds4": "DS4",
    "SCL-32": "S32",
    "SCL-64": "S64",
}
METRIC_TITLES = {
    "ari_overall": "Overall ARI",
    "ari_steadystate": "Steady-State ARI",
    "time_fit_cum_ms": "Learning time [ms]",
    "time_predict_cum_ms": "Prediction time [ms]",
    "memory_max_bytes": "Maximum memory footprint [KiB]",
    "memory_steadystate_bytes": "Steady-state memory footprint [KiB]",
    "mc_max": "Maximum microcluster count",
    "mc_steadystate": "Steady-state microcluster count",
}
CHAIN = ["vectorized", CORRECTED, IMPROVED, FINAL]
PROBE_DATASETS = ["cassini", "chameleon_ds3", "chameleon_ds4"]
QUALITY_METRICS = ["ari_overall", "ari_steadystate"]
COST_METRICS = [
    "mc_max",
    "mc_steadystate",
    "memory_max_bytes",
    "memory_steadystate_bytes",
]
# The experiments whose cost the chapter cites; their cost tables go to its appendix.
APPENDIX_COSTS = {"optimization", "geometry"}
MODEL_NAMES = {"MicroTEDAClus", "MicroTEDAClus+"}
TUNING_FILE = RESULTS_DIR / "tuning.csv"
# Hyperparameters of the sensitivity function that each variant of the m function tunes.
M_PARAMETERS = {
    "m_reparam": (("m_up", r"$m_{\max}$"), ("m_a", r"$S_{95}$")),
    "m_constant": (("m_up", r"$m$"),),
    "m_asymptote": (("m_up", "asymptote"),),
}
PROBE_SPACES = {"final": "MicroTEDAClus+", "engine": "extended"}
CHAIN_COLORS = {
    "original": "xkcd:grey",
    "vectorized": "xkcd:blue",
    CORRECTED: "xkcd:orange",
    IMPROVED: "xkcd:green",
    FINAL: "xkcd:red",
}


# -----------------------------------------------------------------------------
# Formatting
# -----------------------------------------------------------------------------
def model(key: str, stage_names: dict | None = None) -> str:
    """Name of a model or variant in a table: its stage name if it was already accepted, else its label."""
    label = (stage_names or {}).get(key, VARIANTS[key].label)
    macro = "modelname" if label in MODEL_NAMES else "variant"
    return rf"\{macro}{{{label}}}"


def run_set(runs) -> str:
    """The runs of a table, as the chapter names them."""
    if runs == DEVELOPMENT_RUNS:
        return "development runs"
    if runs == VALIDATION_RUNS:
        return "validation runs"
    raise ValueError(f"runs outside the development and validation sets: {runs}")


def number(metric: str, value: float) -> str:
    if value is None or np.isnan(value):
        return "-"
    if metric.startswith("memory"):
        return f"{value / 1024:.0f}"
    if metric.startswith("time"):
        return f"{value:.0f}"
    return METRICS[metric]["fmt"].format(value)


def panel_table(
    folder, name, caption, short, panels, datasets=DATASETS, first_column="Model"
):
    """Panels of rows with one value per dataset, in the folder of an experiment."""
    latex.panel_table(
        ARTIFACT_DIR / folder / f"{name}.tex",
        caption,
        f"tab:mi_{folder}_{name}",
        {d: SHORT_NAMES[d] for d in datasets},
        panels,
        first_column,
        short,
    )


# -----------------------------------------------------------------------------
# Comparison tables (variants in rows, datasets in columns)
# -----------------------------------------------------------------------------
def comparison_table(experiment, metrics, name, caption, short):
    """One panel per metric; each variant is marked against its own control."""
    comparisons = list(experiment.comparisons)
    controls = list(dict.fromkeys(control for _, control in comparisons))
    variants = [variant for variant, _ in comparisons]
    many_controls = len(controls) > 1
    rows = [(c, None) for c in controls if c not in variants] + comparisons

    n_cols = 1 + many_controls + len(DATASETS)
    header = ["Model"] + (["Control"] if many_controls else [])
    header += [SHORT_NAMES[d] for d in DATASETS]
    lines = [" & ".join(header) + r" \\"]

    for metric in metrics:
        lines.append(r"\midrule")
        lines.append(
            rf"\multicolumn{{{n_cols}}}{{l}}{{\textit{{{METRIC_TITLES[metric]}}}}} \\"
        )
        for variant, control in rows:
            cells = [model(variant, experiment.stage_names)]
            if many_controls:
                cells.append(model(control, experiment.stage_names) if control else "")
            values = analysis.medians(variant, metric, experiment.runs)
            verdicts = {}
            if control:
                table = analysis.compare(control, variant, metric, experiment.runs)
                verdicts = table["verdict"].to_dict() if not table.empty else {}
            for dataset in DATASETS:
                value = values.get(dataset, np.nan) if len(values) else np.nan
                cells.append(
                    marked(number(metric, value), verdicts.get(dataset, "same"))
                )
            lines.append(" & ".join(cells) + r" \\")

    spec = "l " + ("l " if many_controls else "") + "c " * len(DATASETS)
    write_table(
        ARTIFACT_DIR / experiment.key / f"{name}.tex",
        caption,
        f"tab:mi_{experiment.key}_{name}",
        spec.strip(),
        lines,
        short,
    )


def experiment_tables(experiment):
    """Clustering quality for the chapter; the cost for its appendix, if the chapter cites it."""
    runs = run_set(experiment.runs)
    comparison_table(
        experiment,
        QUALITY_METRICS,
        "quality",
        f"{experiment.title}: clustering quality, medians over the {runs}.",
        f"{experiment.title}: clustering quality",
    )
    if experiment.key in APPENDIX_COSTS:
        comparison_table(
            experiment,
            ["mc_steadystate", "memory_steadystate_bytes"],
            "cost",
            f"{experiment.title}: steady-state microcluster count and memory footprint, "
            f"medians over the {runs}.",
            f"{experiment.title}: microcluster count and memory footprint",
        )


# -----------------------------------------------------------------------------
# Chain tables (datasets in rows, models in columns, median and quartiles)
# -----------------------------------------------------------------------------
def chain_table(models, metrics, runs, folder, name, caption, short):
    """One panel per metric; each model is marked against the model in the column to its left."""
    header = " & ".join(model(m, STAGE_NAMES) for m in models)
    lines = [f"Dataset & {header}" + r" \\"]
    levels = {m: analysis.run_level(m, runs) for m in models}
    for metric in metrics:
        lines.append(r"\midrule")
        lines.append(
            rf"\multicolumn{{{1 + len(models)}}}{{l}}{{\textit{{{METRIC_TITLES[metric]}}}}} \\"
        )
        for dataset in DATASETS:
            cells = [SHORT_NAMES[dataset]]
            for position, key in enumerate(models):
                rl = levels[key]
                values = rl[rl.dataset == dataset][metric]
                disc = np.asarray(
                    METRICS[metric]["round"](values.to_numpy(dtype=float))
                )
                q1, med, q3 = np.quantile(disc, [0.25, 0.5, 0.75])
                text = f"{number(metric, med)} ({number(metric, q1)}, {number(metric, q3)})"
                verdict = "same"
                if position > 0:
                    table = analysis.compare(models[position - 1], key, metric, runs)
                    verdict = table.loc[dataset, "verdict"]
                cells.append(marked(text, verdict))
            lines.append(" & ".join(cells) + r" \\")
    write_table(
        ARTIFACT_DIR / folder / f"{name}.tex",
        caption,
        f"tab:mi_{folder}_{name}",
        "l " + "c " * len(models),
        lines,
        short,
    )


def validation_tables():
    experiment = EXPERIMENT_BY_KEY["validation"]
    runs = experiment.runs
    note = (
        f"median and quartiles over the {run_set(runs)}. Each model is compared with "
        "the model to its left."
    )
    chain_table(
        CHAIN,
        QUALITY_METRICS,
        runs,
        "validation",
        "quality",
        f"Clustering quality of the accepted models: {note}",
        "Clustering quality of the accepted models",
    )
    chain_table(
        CHAIN,
        COST_METRICS,
        runs,
        "validation",
        "cost",
        f"Microcluster count and memory footprint of the accepted models: {note}",
        "Microcluster count and memory footprint of the accepted models",
    )
    end_to_end = type(experiment)(
        "validation",
        "MicroTEDAClus+ against the vectorized model",
        "",
        ((FINAL, "vectorized"),),
        runs,
        STAGE_NAMES,
    )
    title = r"\MicroTEDAClusPlus{} against the \variant{vectorized} model"
    comparison_table(
        end_to_end,
        QUALITY_METRICS,
        "final_vs_vectorized",
        f"{title}: medians over the {run_set(runs)}.",
        title,
    )


# -----------------------------------------------------------------------------
# Diagnosis of a model (shape of its predicted partitions)
# -----------------------------------------------------------------------------
@lru_cache(maxsize=None)
def true_noise(dataset: str, runs: range) -> float:
    """Share of noise samples in the evaluation segment of a dataset: median over the runs."""
    shares = []
    for run in runs:
        _, y = load_stream(dataset, run)
        shares.append(np.mean(y[int(protocol.SPLIT_FRACTION * len(y)) :] == -1))
    return float(np.median(shares))


def run_medians(variant, runs, column):
    """Median over the runs of one run-level column, per dataset."""
    return analysis.run_level(variant, runs).groupby("dataset")[column].median()


def diagnosis_table(variant, runs, folder="diagnosis"):
    rl = analysis.run_level(variant, runs)
    if rl.empty or "n_pred" not in rl.columns:
        return
    lines = [
        r"Dataset & True clusters & Predicted clusters & Homogeneity & Completeness & "
        r"True noise [\%] & Predicted noise [\%] \\",
        r"\midrule",
    ]
    grouped = rl.groupby("dataset")[
        ["n_true", "n_pred", "homogeneity", "completeness", "noise_pred"]
    ].median()
    for dataset in DATASETS:
        if dataset not in grouped.index:
            continue
        g = grouped.loc[dataset]
        lines.append(
            rf"{SHORT_NAMES[dataset]} & {g.n_true:.1f} & {g.n_pred:.1f} & {g.homogeneity:.2f} & "
            rf"{g.completeness:.2f} & {100 * true_noise(dataset, runs):.1f} & {100 * g.noise_pred:.1f} \\"
        )
    write_table(
        ARTIFACT_DIR / folder / f"{variant}.tex",
        f"Partitions predicted by the {model(variant, STAGE_NAMES)} model: medians over the "
        f"{run_set(runs)} of the means over the batches. The noise label counts as one cluster, and the "
        "first batch, predicted before any sample is learned, is labeled as noise in full, the only "
        "noise of a model without a noise label.",
        f"tab:mi_{folder}_{variant}",
        "l c c c c c c",
        lines,
        f"Partitions predicted by the {model(variant, STAGE_NAMES)} model",
    )


def partition_table(
    folder, name, title, variants, runs=DEVELOPMENT_RUNS, stage_names=None
):
    """Shape of the partitions predicted by several models."""
    labelled = [(model(v, stage_names), v) for v in variants]
    panels = [
        (
            "Predicted clusters",
            [
                (
                    "true clusters",
                    run_medians(variants[0], runs, "n_true"),
                    "{:.1f}".format,
                )
            ]
            + [
                (label, run_medians(v, runs, "n_pred"), "{:.1f}".format)
                for label, v in labelled
            ],
        ),
        (
            "Homogeneity",
            [
                (label, run_medians(v, runs, "homogeneity"), "{:.2f}".format)
                for label, v in labelled
            ],
        ),
        (
            "Completeness",
            [
                (label, run_medians(v, runs, "completeness"), "{:.2f}".format)
                for label, v in labelled
            ],
        ),
    ]
    panel_table(
        folder,
        name,
        f"{title}: predicted partitions, medians over the {run_set(runs)} of the means over the "
        "batches.",
        f"{title}: predicted partitions",
        panels,
    )


def degenerate_table(
    folder, name, title, variants, runs=DEVELOPMENT_RUNS, stage_names=None
):
    """How often a model predicts a single cluster."""
    shares, endings = [], []
    for v in variants:
        rl = analysis.run_level(v, runs).groupby("dataset")
        shares.append(
            (
                model(v, stage_names),
                100 * rl["single_cluster_share"].median(),
                "{:.0f}".format,
            )
        )
        endings.append(
            (model(v, stage_names), rl["ends_single_cluster"].sum(), "{:.0f}".format)
        )
    panel_table(
        folder,
        name,
        f"{title}: predictions of a single cluster. Median share of the batches after the first, and "
        f"number of the {len(runs)} {run_set(runs)} whose penultimate batch holds a single cluster.",
        f"{title}: predictions of a single cluster",
        [
            (r"Batches with a single predicted cluster [\%]", shares),
            ("Runs that end with a single predicted cluster", endings),
        ],
    )


def search_range_table(runs=DEVELOPMENT_RUNS):
    """Variance limits that the wider search range selects, and the degenerate endings they bring."""
    variants = [(model(v, STAGE_NAMES), v) for v in (CORRECTED, "c_nearest_wide")]
    panel_table(
        "search_range",
        "selection",
        "Search range of the variance limit: limit selected, median over the "
        f"{run_set(runs)}, and number of the {len(runs)} runs whose penultimate batch holds a single "
        "predicted cluster.",
        "Search range of the variance limit: selected limits",
        [
            (
                "Variance limit selected",
                [
                    (
                        label,
                        run_medians(v, runs, "param:variance_limit"),
                        "{:.3g}".format,
                    )
                    for label, v in variants
                ],
            ),
            (
                "Runs that end with a single predicted cluster",
                [
                    (
                        label,
                        analysis.run_level(v, runs)
                        .groupby("dataset")["ends_single_cluster"]
                        .sum(),
                        "{:.0f}".format,
                    )
                    for label, v in variants
                ],
            ),
        ],
    )


def noise_label_table(runs=DEVELOPMENT_RUNS):
    """Hyperparameters selected for the noise-label variant and the noise it predicts."""
    noise = {d: 100 * true_noise(d, runs) for d in DATASETS}
    panels = [
        (
            "Variance limit selected, searched up to 100",
            [
                (
                    model(v, STAGE_NAMES),
                    run_medians(v, runs, "param:variance_limit"),
                    "{:.3g}".format,
                )
                for v in (CORRECTED, "i_noise")
            ],
        ),
        (
            "Noise distance selected, searched up to 10",
            [
                (
                    model("i_noise"),
                    run_medians("i_noise", runs, "param:noise_m"),
                    "{:.1f}".format,
                )
            ],
        ),
        (
            r"Samples labelled as noise [\%]",
            [
                ("true noise", noise, "{:.1f}".format),
                (
                    model("i_noise"),
                    100 * run_medians("i_noise", runs, "noise_pred"),
                    "{:.1f}".format,
                ),
            ],
        ),
    ]
    panel_table(
        "improvements_single",
        "noise_label",
        r"Hyperparameters selected for \variant{noise-label} and share of the samples it labels as "
        f"noise: medians over the {run_set(runs)}.",
        r"Hyperparameters and noise of \variant{noise-label}",
        panels,
    )


def m_tuned_table(runs=DEVELOPMENT_RUNS):
    """Values that the search selects for the hyperparameters of the sensitivity function."""
    rows = [
        (
            f"{model(v)}: {name}",
            run_medians(v, runs, f"param:{parameter}"),
            "{:.3g}".format,
        )
        for v, parameters in M_PARAMETERS.items()
        for parameter, name in parameters
    ]
    panel_table(
        "m_function",
        "tuned",
        "Values selected for the hyperparameters of the sensitivity function: medians over the "
        f"{run_set(runs)}.",
        "Hyperparameters selected for the sensitivity function",
        [("Selected value", rows)],
        first_column="Model and hyperparameter",
    )


def tuning_noise_table(runs=DEVELOPMENT_RUNS):
    """Two searches of a single algorithm: the improved and final models in two dimensions."""
    first = analysis.run_level(IMPROVED, runs).set_index(["dataset", "run"])
    second = analysis.run_level(FINAL, runs).set_index(["dataset", "run"])
    parameters = [c for c in first.columns if c.startswith("param:")]
    same_selection = pd.Series(
        np.isclose(first[parameters], second[parameters], rtol=1e-9).all(axis=1),
        index=first.index,
    )
    same_result = pd.Series(
        np.isclose(first["ari_overall"], second["ari_overall"], atol=1e-12),
        index=first.index,
    )
    two_dimensional = [d for d in DATASETS if not d.startswith("SCL-")]
    panel_table(
        "refinements",
        "tuning_noise",
        r"Two hyperparameter searches of a single algorithm with the same seeds, "
        r"\variant{improved} and \variant{improved + no-penalty}, in the two-dimensional datasets: "
        f"number of the {len(runs)} {run_set(runs)}.",
        "Two hyperparameter searches of a single algorithm",
        [
            (
                "Runs",
                [
                    (
                        "both searches select the same hyperparameters",
                        same_selection.groupby("dataset").sum(),
                        "{:.0f}".format,
                    ),
                    (
                        "both evaluations have the same Overall ARI",
                        same_result.groupby("dataset").sum(),
                        "{:.0f}".format,
                    ),
                ],
            )
        ],
        datasets=two_dimensional,
        first_column="",
    )


def trials_table(runs=DEVELOPMENT_RUNS):
    """Budget that the hyperparameter search spends on each implementation."""
    tuning = pd.read_csv(TUNING_FILE)
    tuning = tuning[
        tuning["variant"].isin(["original", "vectorized"])
        & tuning["run"].isin(runs)
        & (tuning["status"] == "OK")
    ]
    tuning = tuning.drop_duplicates(["variant", "dataset", "run"], keep="last")
    for column in ("trials", "tuning_seconds"):
        tuning[column] = pd.to_numeric(tuning[column])
    grouped = {
        v: tuning[tuning["variant"] == v].groupby("dataset")
        for v in ("original", "vectorized")
    }
    panel_table(
        "optimization",
        "trials",
        f"Hyperparameter search of the two implementations: medians over the {run_set(runs)}. The "
        f"search ends at {protocol.OPT_TRIALS} trials or {protocol.OPT_TIMEOUT} seconds.",
        "Hyperparameter search of the two implementations",
        [
            (
                "Completed trials",
                [
                    (model(v), g["trials"].median(), "{:.0f}".format)
                    for v, g in grouped.items()
                ],
            ),
            (
                "Duration of the search [s]",
                [
                    (model(v), g["tuning_seconds"].median(), "{:.0f}".format)
                    for v, g in grouped.items()
                ],
            ),
        ],
    )


# -----------------------------------------------------------------------------
# Serial timing (timing.py)
# -----------------------------------------------------------------------------
def serial_times() -> pd.DataFrame:
    """Times of each task of the serial pass: the minimum over its repetitions."""
    t = pd.read_csv(TIMING_FILE)
    t["time_total_ms"] = t["time_fit_ms"] + t["time_predict_ms"]
    return t.groupby(
        ["dataset", "run", "model", "hyperparameters"], as_index=False
    ).agg(
        time_fit_ms=("time_fit_ms", "min"),
        time_predict_ms=("time_predict_ms", "min"),
        time_total_ms=("time_total_ms", "min"),
        microclusters=("microclusters", "first"),
    )


def repetition_spread() -> float:
    """Median over the tasks of the gap between their slowest and fastest executions, in percent."""
    t = pd.read_csv(TIMING_FILE)
    total = (t["time_fit_ms"] + t["time_predict_ms"]).groupby(
        [t["dataset"], t["run"], t["model"], t["hyperparameters"]]
    )
    return float(100 * (total.max() / total.min() - 1).median())


def quartiles(values, fmt) -> str:
    q1, med, q3 = np.quantile(np.asarray(values, dtype=float), [0.25, 0.5, 0.75])
    return f"{fmt(med)} ({fmt(q1)}, {fmt(q3)})"


def milliseconds(value: float) -> str:
    return f"{value:.0f}" if value >= 100 else f"{value:.1f}"


def speedup_table():
    """Original against vectorized implementation, both with the same hyperparameters."""
    times = serial_times()
    lines = [
        r"Dataset & \variant{original} [ms] & \variant{vectorized} [ms] & Speedup \\"
    ]
    for source in ("original", "vectorized"):
        shared = times[times.hyperparameters == source].pivot(
            index=["dataset", "run"], columns="model", values="time_total_ms"
        )
        shared["speedup"] = shared["original"] / shared["vectorized"]
        lines.append(r"\midrule")
        lines.append(
            rf"\multicolumn{{4}}{{l}}{{\textit{{Hyperparameters tuned for the {model(source)} model}}}} \\"
        )
        groups = [(SHORT_NAMES[d], shared.loc[d]) for d in DATASETS]
        groups.append(("All", shared))
        for name, rows in groups:
            lines.append(
                " & ".join(
                    [
                        name,
                        quartiles(rows["original"], milliseconds),
                        quartiles(rows["vectorized"], milliseconds),
                        quartiles(rows["speedup"], "{:.1f}".format),
                    ]
                )
                + r" \\"
            )
    runs = times["run"].nunique()
    write_table(
        ARTIFACT_DIR / "optimization" / "speedup.tex",
        "Execution time of the evaluation segment, learning plus prediction, of the two "
        "implementations with the same hyperparameters, and speedup of the vectorized one: median and "
        f"quartiles over the {runs} development runs. Each task runs alone, and its time is the "
        f"smallest of three executions, which differ by {repetition_spread():.1f}\\% in the median.",
        "tab:mi_optimization_speedup",
        "l c c c",
        lines,
        "Execution time of the two implementations with the same hyperparameters",
    )


def speedup_figure():
    """Speedup of each run against the number of microclusters of its configuration."""
    times = serial_times()
    colors = plt.get_cmap("tab10")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True, sharey=True)
    for ax, source in zip(axes, ("original", "vectorized")):
        shared = times[times.hyperparameters == source]
        total = shared.pivot(
            index=["dataset", "run"], columns="model", values="time_total_ms"
        )
        count = shared[shared.model == "vectorized"].set_index(["dataset", "run"])[
            "microclusters"
        ]
        for position, dataset in enumerate(DATASETS):
            ax.scatter(
                count.loc[dataset].clip(lower=1),
                (total["original"] / total["vectorized"]).loc[dataset],
                s=14,
                color=colors(position),
                label=SHORT_NAMES[dataset],
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_yticks([5, 10, 20, 50, 100])
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_title(
            f"Hyperparameters tuned for the {VARIANTS[source].label} model", fontsize=10
        )
        ax.set_xlabel("Microclusters at the penultimate batch")
    axes[0].set_ylabel("Speedup")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(DATASETS),
        bbox_to_anchor=(0.5, 1.05),
    )
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    out = ARTIFACT_DIR / "optimization" / "plots" / "speedup.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


def serial_cost_table():
    """Cost of the original implementation and of the accepted models, each as tuned for its runs."""
    times = serial_times()
    own = times[times.model == times.hyperparameters]
    models = ["original", *CHAIN]
    wide = {
        column: own.pivot(index=["dataset", "run"], columns="model", values=column)
        for column in (
            "time_fit_ms",
            "time_predict_ms",
            "time_total_ms",
            "microclusters",
        )
    }
    relative = wide["time_total_ms"].div(wide["time_total_ms"]["vectorized"], axis=0)
    panels = [
        ("Learning time [ms]", wide["time_fit_ms"], milliseconds),
        ("Prediction time [ms]", wide["time_predict_ms"], milliseconds),
        (
            r"Total time relative to the \variant{vectorized} model, per run",
            relative,
            "{:.2f}".format,
        ),
        ("Steady-state microcluster count", wide["microclusters"], "{:.0f}".format),
    ]
    header = " & ".join(model(m, STAGE_NAMES) for m in models)
    lines = [f"Dataset & {header}" + r" \\"]
    for title, table, fmt in panels:
        lines.append(r"\midrule")
        lines.append(
            rf"\multicolumn{{{1 + len(models)}}}{{l}}{{\textit{{{title}}}}} \\"
        )
        for dataset in DATASETS:
            cells = [SHORT_NAMES[dataset]]
            cells += [quartiles(table.loc[dataset][m].dropna(), fmt) for m in models]
            lines.append(" & ".join(cells) + r" \\")
    runs = times["run"].nunique()
    write_table(
        ARTIFACT_DIR / "validation" / "serial_cost.tex",
        "Cost of the original implementation and of the accepted models on the evaluation segment, each "
        f"with its own tuned hyperparameters: median and quartiles over the {runs} development runs. "
        "Each task runs alone, and its time is the smallest of three executions, which differ by "
        f"{repetition_spread():.1f}\\% in the median.",
        "tab:mi_validation_serial_cost",
        "l " + "c " * len(models),
        lines,
        "Cost of the original implementation and of the accepted models",
    )


# -----------------------------------------------------------------------------
# Reachability probe (probe.py)
# -----------------------------------------------------------------------------
def probe_configurations(space: str, dataset: str) -> pd.DataFrame:
    """One row per probed configuration: its means over the probed streams."""
    df = pd.read_csv(PROBE_DIR / f"{space}_{dataset}.csv")
    complete = df.groupby("configuration").filter(
        lambda rows: len(rows) == df["run"].nunique()
    )
    return complete.groupby("configuration").agg(
        tuning_objective=("tuning_objective", "mean"),
        ari_overall=("ari_overall", "mean"),
        ari_steadystate=("ari_steadystate", "mean"),
        microclusters=("microclusters", "median"),
        variance_limit=("variance_limit", "first"),
        decay=("decay", "first"),
    )


def probe_table():
    """What the search space holds against what the tuning objective selects."""
    lines = [
        r"Dataset & Space & Configuration & Objective & Overall ARI & Steady-State ARI & "
        r"Microclusters & $r$ & decay \\"
    ]
    summary = []
    for dataset in PROBE_DATASETS:
        lines.append(r"\midrule")
        tuned = analysis.run_level(FINAL, DEVELOPMENT_RUNS)
        tuned = tuned[tuned.dataset == dataset]
        protocol_overall = tuned["ari_overall"].median()
        lines.append(
            rf"{SHORT_NAMES[dataset]} & final & tuned by the protocol & - & {protocol_overall:.3f} & "
            rf"{tuned['ari_steadystate'].median():.3f} & {tuned['mc_steadystate'].median():.0f} & "
            rf"{tuned['param:variance_limit'].median():.4f} & {tuned['param:decay'].median():.0f} \\"
        )
        for space, space_name in PROBE_SPACES.items():
            configurations = probe_configurations(space, dataset)
            picks = {
                "highest objective": configurations["tuning_objective"].idxmax(),
                "highest Overall ARI": configurations["ari_overall"].idxmax(),
            }
            for name, index in picks.items():
                c = configurations.loc[index]
                lines.append(
                    rf" & {space_name} & {name} & {c.tuning_objective:.3f} & {c.ari_overall:.3f} & "
                    rf"{c.ari_steadystate:.3f} & {c.microclusters:.0f} & {c.variance_limit:.4f} & "
                    rf"{c.decay:.0f} \\"
                )
            above = configurations[configurations.ari_overall > protocol_overall]
            summary.append(
                {
                    "dataset": dataset,
                    "space": space_name,
                    "configurations": len(configurations),
                    "above_protocol": len(above),
                    "above_protocol_over_100_microclusters": int(
                        (above.microclusters > 100).sum()
                    ),
                    "spearman": configurations["tuning_objective"].corr(
                        configurations["ari_overall"], method="spearman"
                    ),
                }
            )
    write_table(
        ARTIFACT_DIR / "selection" / "probe.tex",
        "Reachability probe: for each search space, the configuration of highest objective and the one "
        "of highest Overall ARI. The first row of each dataset is \\MicroTEDAClusPlus{} as the protocol "
        "tunes it, with medians over the development runs; the microcluster count is that of the "
        "penultimate batch.",
        "tab:mi_selection_probe",
        "l l l c c c c c c",
        lines,
        "Reachability probe of two search spaces",
    )
    pd.DataFrame(summary).to_csv(PROBE_DIR / "summary.csv", index=False)


def probe_figure():
    """Tuning objective against Overall ARI of the configurations of the space of MicroTEDAClus+."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    tuned = analysis.run_level(FINAL, DEVELOPMENT_RUNS)
    for ax, dataset in zip(axes, PROBE_DATASETS):
        c = probe_configurations("final", dataset)
        points = ax.scatter(
            c["tuning_objective"],
            c["ari_overall"],
            c=np.log10(c["microclusters"].clip(lower=1)),
            cmap="viridis",
            vmin=0,
            vmax=3.2,
            s=14,
        )
        for index, marker, label in (
            (c["tuning_objective"].idxmax(), "s", "Highest objective"),
            (c["ari_overall"].idxmax(), "^", "Highest Overall ARI"),
        ):
            ax.scatter(
                c.loc[index, "tuning_objective"],
                c.loc[index, "ari_overall"],
                marker=marker,
                s=110,
                facecolors="none",
                edgecolors="xkcd:red",
                linewidths=1.8,
                label=label,
            )
        ax.axhline(
            tuned[tuned.dataset == dataset]["ari_overall"].median(),
            color="xkcd:grey",
            linestyle="--",
            linewidth=1.2,
            label="Tuned by the protocol (median)",
        )
        ax.set_title(dataset_display_name(dataset), fontsize=10)
        ax.set_xlabel("Tuning objective")
    axes[0].set_ylabel("Overall ARI")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.04))
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    bar = fig.colorbar(points, ax=axes, pad=0.01, ticks=[0, 1, 2, 3])
    bar.ax.set_yticklabels(["1", "10", "100", "1000"])
    bar.set_label("Microclusters")
    out = ARTIFACT_DIR / "selection" / "plots" / "probe.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


def maturation_table():
    """Quality and structure along the stream of the two configurations singled out by the probe."""
    df = pd.read_csv(PROBE_DIR / "maturation.csv")
    segments = sorted(int(c.split("_")[1]) for c in df.columns if c.startswith("ari_"))
    means = df.groupby(["dataset", "configuration"], sort=False).mean(numeric_only=True)
    width = 100 // len(segments)
    spans = [f"{(i - 1) * width}--{i * width}\\%" for i in segments]
    lines = ["Dataset & Configuration & " + " & ".join(spans) + r" \\"]
    panels = [
        ("Median batch ARI", "ari", "{:.3f}".format),
        ("Median number of predicted clusters", "clusters", "{:.1f}".format),
        ("Microclusters at the end of the segment", "microclusters", "{:.0f}".format),
    ]
    for title, column, fmt in panels:
        lines.append(r"\midrule")
        lines.append(
            rf"\multicolumn{{{2 + len(segments)}}}{{l}}{{\textit{{{title}}}}} \\"
        )
        for dataset in PROBE_DATASETS:
            rows = list(means.loc[dataset].iterrows())
            if column == "clusters":
                first = rows[0][1]
                true = pd.Series(
                    {f"clusters_{i}": first[f"true_clusters_{i}"] for i in segments}
                )
                rows.insert(0, ("true clusters", true))
            for position, (name, row) in enumerate(rows):
                cells = [SHORT_NAMES[dataset] if position == 0 else "", name]
                cells += [fmt(row[f"{column}_{i}"]) for i in segments]
                lines.append(" & ".join(cells) + r" \\")
    write_table(
        ARTIFACT_DIR / "selection" / "maturation.tex",
        "The two configurations of the search space of \\MicroTEDAClusPlus{} in "
        "\\cref{tab:mi_selection_probe} along the stream, in fifths, when the model learns the whole "
        "stream from its first sample. The first fifth is the segment that the hyperparameter search sees.",
        "tab:mi_selection_maturation",
        "l l " + "c " * len(segments),
        lines,
        "The two probed configurations of \\MicroTEDAClusPlus{} along the stream",
    )


# -----------------------------------------------------------------------------
# Main experiment: the final model against the baseline, per dimension
# -----------------------------------------------------------------------------
def trajectories(models, runs, name, folder="validation"):
    """Median and interquartile range of the batch ARI of each model, per dataset."""
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.5), sharey=True)
    for ax, dataset in zip(axes.ravel(), DATASETS):
        for key in models:
            curves = []
            for run in runs:
                output_file, _ = result_paths(key, dataset, run)
                if not output_file.exists():
                    continue
                df = pd.read_csv(output_file)
                if "status" in df.columns and is_valid_result(df):
                    curves.append(add_time_columns(df)["ari"].to_numpy())
            if not curves:
                continue
            length = min(len(c) for c in curves)
            stack = np.vstack([c[:length] for c in curves])
            batches = np.arange(length)
            q1, med, q3 = np.quantile(stack, [0.25, 0.5, 0.75], axis=0)
            ax.plot(
                batches,
                med,
                color=CHAIN_COLORS[key],
                linewidth=1.5,
                label=STAGE_NAMES.get(key, VARIANTS[key].label),
            )
            ax.fill_between(batches, q1, q3, color=CHAIN_COLORS[key], alpha=0.15)
        ax.set_title(dataset_display_name(dataset), fontsize=10)
        ax.set_xlabel("Evaluation Batch")
    for ax in axes[:, 0]:
        ax.set_ylabel("ARI")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(models),
        bbox_to_anchor=(0.5, 1.02),
    )
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    out = ARTIFACT_DIR / folder / "plots" / f"{name}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Decisions
# -----------------------------------------------------------------------------
def available(experiment) -> bool:
    return all(
        not analysis.run_level(v, experiment.runs).empty for v in experiment.variants
    )


def decisions() -> pd.DataFrame:
    """Outcome of the decision rule for every comparison that has results."""
    rows = []
    for experiment in EXPERIMENTS:
        if not available(experiment):
            continue
        for variant, control in experiment.comparisons:
            d = analysis.decide(control, variant, experiment.runs)
            row = {
                "experiment": experiment.key,
                "variant": variant,
                "control": control,
                "outcome": d["outcome"],
            }
            for metric in analysis.DECISION_METRICS:
                short = metric.replace("ari_", "")
                row[f"{short}_better"] = " ".join(
                    SHORT_NAMES[x] for x in d["better"].get(metric, [])
                )
                row[f"{short}_worse"] = " ".join(
                    SHORT_NAMES[x] for x in d["worse"].get(metric, [])
                )
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--summary", action="store_true", help="only print the decisions"
    )
    args = parser.parse_args()

    table = decisions()
    if not table.empty:
        table.to_csv(RESULTS_DIR / "decisions.csv", index=False)
    if args.summary:
        pd.set_option(
            "display.width", 250, "display.max_rows", 200, "display.max_colwidth", 40
        )
        print(table.to_string(index=False))
        return

    for experiment in EXPERIMENTS:
        if not available(experiment):
            print(f"{experiment.key}: results missing, tables skipped")
            continue
        if experiment.key == "validation":
            validation_tables()
            trajectories(CHAIN, VALIDATION_RUNS, "ari_trajectories")
        else:
            experiment_tables(experiment)
        print(f"{experiment.key}: tables written")
    diagnosis_table(CORRECTED, DEVELOPMENT_RUNS)
    degenerate_table(
        "corrections_single",
        "degenerate",
        "Corrections, one at a time",
        ["vectorized", "c_nopen", "c_nearest"],
    )
    partition_table(
        "corrections_combined",
        "partitions",
        "Combined corrections",
        ["c_nearest", "cc_nearest_nopen"],
    )
    degenerate_table(
        "improvements_single",
        "degenerate",
        "Improvements, one at a time",
        [CORRECTED, "i_relative", "i_chi"],
        stage_names=STAGE_NAMES,
    )
    search_range_table()
    noise_label_table()
    m_tuned_table()
    tuning_noise_table()
    if TUNING_FILE.exists():
        trials_table()

    if TIMING_FILE.exists():
        speedup_table()
        speedup_figure()
        serial_cost_table()
        print("serial timing: tables written")
    if all(
        (PROBE_DIR / f"{s}_{d}.csv").exists()
        for s in PROBE_SPACES
        for d in PROBE_DATASETS
    ):
        probe_table()
        probe_figure()
        print("probe: table and figure written")
    if (PROBE_DIR / "maturation.csv").exists():
        maturation_table()
        print("probe: maturation table written")


if __name__ == "__main__":
    main()
