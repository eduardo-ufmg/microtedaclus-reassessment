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
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from compute_metrics import add_time_columns, is_valid_result, scalar_metrics
from editorial_config import dataset_display_name, model_display_name
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
from statistic_analysis import METRICS

ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = ROOT.parent.parent / "monografia" / "ModelImprovement" / "artifacts"
TIMING_FILE = RESULTS_DIR / "_timing" / "timing.csv"
PROBE_DIR = RESULTS_DIR / "_probe"
MAIN_SCALABILITY_FILE = ROOT / "metrics" / "scalability" / "run_level.parquet"

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
    "ari_fifth": "ARI at the Fifth Batch",
    "ari_diff": "Maximum ARI Drop",
    "time_fit_cum_ms": "Cumulative Fitting Time [ms]",
    "time_predict_cum_ms": "Cumulative Prediction Time [ms]",
    "memory_max_bytes": "Maximum Memory Footprint [kB]",
    "memory_steadystate_bytes": "Steady-State Memory Footprint [kB]",
    "mc_max": "Maximum Microcluster Count",
    "mc_steadystate": "Steady-State Microcluster Count",
}
OUTCOME_SYMBOLS = {
    "improves": "$+$",
    "regresses": "$-$",
    "mixed": r"$\pm$",
    "indistinguishable": "$=$",
    "missing": "",
}
CHAIN = ["vectorized", CORRECTED, IMPROVED, FINAL]
PROBE_DATASETS = ["cassini", "chameleon_ds3", "chameleon_ds4"]
PROBE_SPACES = {"final": "final", "engine": "extended"}
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
    """Name of a model in a table: its stage name if it was already accepted, else its label."""
    label = (stage_names or {}).get(key, VARIANTS[key].label)
    return rf"\modelname{{{label}}}"


def number(metric: str, value: float) -> str:
    if value is None or np.isnan(value):
        return "-"
    if metric.startswith("memory"):
        return f"{value / 1024:.0f}"
    if metric.startswith("time"):
        return f"{value:.0f}"
    return METRICS[metric]["fmt"].format(value)


def marked(text: str, verdict: str) -> str:
    if verdict == "better":
        return rf"\textbf{{{text}}}"
    if verdict == "worse":
        return rf"\textcolor{{red}}{{{text}}}"
    return text


def write_table(
    path: Path, caption: str, label: str, column_spec: str, lines: list[str]
):
    path.parent.mkdir(parents=True, exist_ok=True)
    body = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{caption}}}",
        r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{{column_spec}}}",
        r"\toprule",
        *lines,
        r"\bottomrule",
        r"\end{tabular}",
        "}",
        rf"\label{{{label}}}",
        r"\end{table}",
    ]
    path.write_text("\n".join(body) + "\n")


# -----------------------------------------------------------------------------
# Comparison tables (variants in rows, datasets in columns)
# -----------------------------------------------------------------------------
def comparison_table(experiment, metrics, name, caption, with_rule=True):
    """One panel per metric; each variant is marked against its own control."""
    comparisons = list(experiment.comparisons)
    controls = list(dict.fromkeys(control for _, control in comparisons))
    variants = [variant for variant, _ in comparisons]
    many_controls = len(controls) > 1
    rows = [(c, None) for c in controls if c not in variants] + comparisons

    n_cols = 1 + many_controls + len(DATASETS) + with_rule
    header = ["Model"] + (["Control"] if many_controls else [])
    header += [SHORT_NAMES[d] for d in DATASETS] + (["Rule"] if with_rule else [])
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
            if with_rule:
                outcome = (
                    analysis.decide(control, variant, experiment.runs)["outcome"]
                    if control
                    else ""
                )
                cells.append(OUTCOME_SYMBOLS.get(outcome, ""))
            lines.append(" & ".join(cells) + r" \\")

    spec = "l " + ("l " if many_controls else "") + "c " * (len(DATASETS) + with_rule)
    write_table(
        ARTIFACT_DIR / experiment.key / f"{name}.tex",
        caption,
        f"tab:mi_{experiment.key}_{name}",
        spec.strip(),
        lines,
    )


def experiment_tables(experiment):
    """Decision metrics for the chapter; the remaining metrics for the appendix."""
    runs = f"runs {experiment.runs.start} to {experiment.runs.stop - 1}"
    marks = ""
    comparison_table(
        experiment,
        ["ari_overall", "ari_steadystate"],
        "quality",
        f"{experiment.title}: Overall and Steady-State ARI. {marks}. The last column gives the outcome "
        "of the decision rule: improvement ($+$), regression ($-$), both ($\\pm$) or neither ($=$).",
    )
    comparison_table(
        experiment,
        ["ari_fifth", "ari_diff"],
        "quality_other",
        f"{experiment.title}: ARI at the fifth batch and maximum ARI drop. {marks}.",
        with_rule=False,
    )
    comparison_table(
        experiment,
        ["mc_steadystate", "memory_steadystate_bytes"],
        "cost",
        f"{experiment.title}: steady-state microcluster count and memory footprint. {marks}.",
        with_rule=False,
    )
    comparison_table(
        experiment,
        ["time_fit_cum_ms", "time_predict_cum_ms"],
        "time",
        f"{experiment.title}: cumulative fitting and prediction times, measured under parallel load. "
        f"{marks}.",
        with_rule=False,
    )


def variants_table(experiment):
    """What each variant of an experiment changes."""
    lines = [r"Model & Change relative to its base \\", r"\midrule"]
    for key in experiment.variants:
        variant = VARIANTS[key]
        if variant.summary:
            lines.append(
                rf"{model(key, experiment.stage_names)} & {variant.summary} \\"
            )
    path = ARTIFACT_DIR / experiment.key / "variants.tex"
    path.parent.mkdir(parents=True, exist_ok=True)
    body = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{experiment.title}: models evaluated.}}",
        r"\begin{tabular}{l p{0.62\textwidth}}",
        r"\toprule",
        *lines,
        r"\bottomrule",
        r"\end{tabular}",
        rf"\label{{tab:mi_{experiment.key}_variants}}",
        r"\end{table}",
    ]
    path.write_text("\n".join(body) + "\n")


# -----------------------------------------------------------------------------
# Chain tables (datasets in rows, models in columns, median and quartiles)
# -----------------------------------------------------------------------------
def chain_table(models, metric, runs, folder, name, caption):
    """Each model is marked against the model in the column to its left."""
    header = " & ".join(model(m, STAGE_NAMES) for m in models)
    lines = [f"Dataset & {header}" + r" \\", r"\midrule"]
    levels = {m: analysis.run_level(m, runs) for m in models}
    for dataset in DATASETS:
        cells = [rf"\datasetname{{{dataset_display_name(dataset)}}}"]
        for position, key in enumerate(models):
            rl = levels[key]
            values = (
                rl[rl.dataset == dataset][metric]
                if not rl.empty
                else pd.Series(dtype=float)
            )
            if values.empty:
                cells.append("-")
                continue
            disc = np.asarray(METRICS[metric]["round"](values.to_numpy(dtype=float)))
            q1, med, q3 = np.quantile(disc, [0.25, 0.5, 0.75])
            text = f"{number(metric, med)} ({number(metric, q1)}, {number(metric, q3)})"
            verdict = "same"
            if position > 0:
                table = analysis.compare(models[position - 1], key, metric, runs)
                if not table.empty and dataset in table.index:
                    verdict = table.loc[dataset, "verdict"]
            cells.append(marked(text, verdict))
        lines.append(" & ".join(cells) + r" \\")
    write_table(
        ARTIFACT_DIR / folder / f"{name}.tex",
        caption,
        f"tab:mi_{folder}_{name}",
        "l " + "c " * len(models),
        lines,
    )


def validation_tables():
    experiment = EXPERIMENT_BY_KEY["validation"]
    runs = experiment.runs
    note = (
        f"Median and quartiles over the {len(runs)} validation runs. Each model is compared with the "
        "model to its left: bold is significantly better, red is significantly worse."
    )
    for metric in METRIC_TITLES:
        chain_table(
            CHAIN,
            metric,
            runs,
            "validation",
            metric,
            f"{METRIC_TITLES[metric]} of the accepted models. {note}",
        )
    end_to_end = type(experiment)(
        "validation",
        "Final model against the vectorized model",
        "",
        ((FINAL, "vectorized"),),
        runs,
        STAGE_NAMES,
    )
    comparison_table(
        end_to_end,
        ["ari_overall", "ari_steadystate", "ari_fifth", "ari_diff"],
        "final_vs_vectorized",
        f"Final model against the vectorized model on the {len(runs)} validation runs. Bold and red "
        "values are significantly better and worse than the vectorized model, respectively.",
        with_rule=False,
    )


# -----------------------------------------------------------------------------
# Diagnosis of a model (shape of its predicted partitions)
# -----------------------------------------------------------------------------
def diagnosis_table(variant, runs, folder="diagnosis"):
    rl = analysis.run_level(variant, runs)
    if rl.empty or "n_pred" not in rl.columns:
        return
    lines = [
        r"Dataset & True clusters & Predicted clusters & Homogeneity & Completeness & Predicted noise [\%] \\",
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
            rf"\datasetname{{{dataset_display_name(dataset)}}} & {g.n_true:.1f} & {g.n_pred:.1f} & "
            rf"{g.homogeneity:.2f} & {g.completeness:.2f} & {100 * g.noise_pred:.1f} \\"
        )
    write_table(
        ARTIFACT_DIR / folder / f"{variant}.tex",
        f"Predicted partitions of the {model(variant, STAGE_NAMES)} model: medians over the development "
        "runs of the per-run means over the batches. The noise label counts as one cluster in the batches "
        "that contain it. The first batch is predicted before any sample is learned and is labelled as noise "
        "in full, which accounts for the predicted noise of a model that never labels noise.",
        f"tab:mi_{folder}_{variant}",
        "l c c c c c",
        lines,
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


def quartiles(values, fmt) -> str:
    q1, med, q3 = np.quantile(np.asarray(values, dtype=float), [0.25, 0.5, 0.75])
    return f"{fmt(med)} ({fmt(q1)}, {fmt(q3)})"


def milliseconds(value: float) -> str:
    return f"{value:.0f}" if value >= 100 else f"{value:.1f}"


def speedup_table():
    """Original against vectorized implementation, both with the same hyperparameters."""
    times = serial_times()
    lines = [
        r"Dataset & \modelname{original} [ms] & \modelname{vectorized} [ms] & Speedup \\"
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
        "Execution time of the evaluation segment, fitting plus prediction, of the original and "
        "vectorized implementations with the same hyperparameters, and speedup of the latter, computed "
        f"per run: median and quartiles over the {runs} development runs. Each task is executed alone, "
        "and its time is the minimum of three executions.",
        "tab:mi_optimization_speedup",
        "l c c c",
        lines,
    )


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
        ("Fitting time [ms]", wide["time_fit_ms"], milliseconds),
        ("Prediction time [ms]", wide["time_predict_ms"], milliseconds),
        (
            r"Total time relative to the \modelname{vectorized} model, per run",
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
        f"with the hyperparameters tuned for it: median and quartiles over the {runs} development runs. "
        "Each task is executed alone, and its time is the minimum of three executions.",
        "tab:mi_validation_serial_cost",
        "l " + "c " * len(models),
        lines,
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
        "Configurations of two search spaces scored by the objective of the hyperparameter search, on the "
        "first 20\\% of the stream, and by the evaluation metrics, on the remaining 80\\%. For each space, "
        "400 configurations are drawn at random, and the rows give the one with the highest objective and "
        "the one with the highest Overall ARI. The values are means over four streams in CSN and the "
        "values of the single stream in DS3 and DS4; the microcluster count is that of the penultimate "
        "batch. The first row of each dataset is the final model as the protocol tunes it: medians over "
        "the development runs.",
        "tab:mi_selection_probe",
        "l l l c c c c c c",
        lines,
    )
    pd.DataFrame(summary).to_csv(PROBE_DIR / "summary.csv", index=False)


def probe_figure():
    """Tuning objective against Overall ARI of the configurations of the final model's space."""
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
    """Batch ARI along the stream of the two configurations singled out by the probe."""
    df = pd.read_csv(PROBE_DIR / "maturation.csv")
    segments = [c for c in df.columns if c.startswith("ari_")]
    means = df.groupby(["dataset", "configuration"], sort=False)[
        [*segments, "microclusters_first_segment"]
    ].mean()
    width = 100 // len(segments)
    spans = [f"{i * width}--{(i + 1) * width}\\%" for i in range(len(segments))]
    lines = [
        "Dataset & Configuration & "
        + " & ".join(spans)
        + r" & Microclusters at 20\% \\"
    ]
    for dataset in PROBE_DATASETS:
        lines.append(r"\midrule")
        for position, (name, row) in enumerate(means.loc[dataset].iterrows()):
            cells = [SHORT_NAMES[dataset] if position == 0 else "", name]
            cells += [f"{row[c]:.3f}" for c in segments]
            cells.append(f"{row['microclusters_first_segment']:.0f}")
            lines.append(" & ".join(cells) + r" \\")
    write_table(
        ARTIFACT_DIR / "selection" / "maturation.tex",
        "Median batch ARI in each fifth of the stream, for the two configurations of the final model's "
        "space of \\cref{tab:mi_selection_probe}, when the model learns the whole stream from its first "
        "sample. The first fifth is the segment that the hyperparameter search sees. Means over four "
        "streams in CSN; single stream in DS3 and DS4.",
        "tab:mi_selection_maturation",
        "l l " + "c " * (len(segments) + 1),
        lines,
    )


# -----------------------------------------------------------------------------
# Main experiment: the final model against the baseline, per dimension
# -----------------------------------------------------------------------------
def main_scalability_table(baseline="MicroTEDAClus", proposed="MicroTEDAClusPlus"):
    """ARI metrics of the scalability streams of the main experiment, per dimension."""
    batches = pd.read_parquet(MAIN_SCALABILITY_FILE)
    batches = batches[batches.model.isin([baseline, proposed])]
    runs = pd.DataFrame(
        [
            {
                "dim": dim,
                "model": name,
                "run": run,
                **scalar_metrics(g.sort_values("batch")),
            }
            for (dim, name, run), g in batches.groupby(["dim", "model", "run"])
        ]
    )
    dims = sorted(runs["dim"].unique())
    metrics = ["ari_overall", "ari_steadystate", "ari_fifth", "ari_diff"]
    lines = ["Model & " + " & ".join(f"$d = {d}$" for d in dims) + r" \\"]
    for metric in metrics:
        wide = runs.pivot(index=["dim", "run"], columns="model", values=metric).dropna()
        pairs = {d: (wide.loc[d][baseline], wide.loc[d][proposed]) for d in dims}
        table = analysis.paired_comparisons(pairs, metric)
        lines.append(r"\midrule")
        lines.append(
            rf"\multicolumn{{{1 + len(dims)}}}{{l}}{{\textit{{{METRIC_TITLES[metric]}}}}} \\"
        )
        lines.append(
            " & ".join(
                [rf"\modelname{{{model_display_name(baseline)}}}"]
                + [number(metric, table.loc[d, "control_median"]) for d in dims]
            )
            + r" \\"
        )
        lines.append(
            " & ".join(
                [rf"\modelname{{{model_display_name(proposed)}}}"]
                + [
                    marked(
                        number(metric, table.loc[d, "median"]), table.loc[d, "verdict"]
                    )
                    for d in dims
                ]
            )
            + r" \\"
        )
    write_table(
        ARTIFACT_DIR / "main_experiment" / "scalability_ari.tex",
        "ARI metrics of the scalability streams of the main experiment, per dimension. "
        "$p$-values are adjusted over the seven dimensions.",
        "tab:mi_main_scalability_ari",
        "l " + "c " * len(dims),
        lines,
    )


# -----------------------------------------------------------------------------
# Figures
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
        variants_table(experiment)
        if not available(experiment):
            print(f"{experiment.key}: results missing, tables skipped")
            continue
        if experiment.key == "validation":
            validation_tables()
            trajectories(CHAIN, VALIDATION_RUNS, "ari_trajectories")
        else:
            experiment_tables(experiment)
        print(f"{experiment.key}: tables written")
    for variant in ("vectorized", CORRECTED):
        diagnosis_table(variant, DEVELOPMENT_RUNS)

    if TIMING_FILE.exists():
        speedup_table()
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
    if MAIN_SCALABILITY_FILE.exists():
        main_scalability_table()
        print("main experiment: scalability table written")


if __name__ == "__main__":
    main()
