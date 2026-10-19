"""Tables of the chapter on the improvement of MicroTEDAClus+ on the real datasets.

Written under monografia/RealImprovement/artifacts/, each from its sources, and
only when they exist:

- chain.tex: the kept changes, each against the step before it, on the
  development runs (results_improvement);
- selection.tex: the selected variance limit and link factor along the chain;
- candidates.tex: every candidate outside the chain against its control;
- diagnosis.tex: what MicroTEDAClus+ of Chapter 3 does on the real datasets, from
  the replay of its selected configurations (results_real/diagnosis);
- confirmation.tex: the chain on the runs of the real-data experiment
  (results_confirmation, with MicroTEDAClus+ from results_real);
- quality.tex, failures.tex and cost.tex: the final version of MicroTEDAClus+,
  tuned and evaluated as every model of the real-data experiment
  (results_final_model, results_timing_real, results_memory_real), against the
  benchmark models of that experiment;
- partitions.tex: the partitions of the two versions of MicroTEDAClus+, from the
  replays of their selected configurations.

Usage (from experimentos/modern_methodology):
    python -m benchmark.improvement.artifacts
"""

import numpy as np
import pandas as pd

from benchmark import analysis
from benchmark.artifacts import (
    GROUPS,
    TITLES,
    _in_kib,
    _in_ms,
    at_the_ends,
    tier_panels,
)
from benchmark.display import columns
from benchmark.hyperparameters import Category, widened_ranges_of_categories
from benchmark.improvement.candidates import CANDIDATES
from benchmark.improvement.compare import comparison, outcome
from benchmark.improvement.run import DATASETS as DEVELOPMENT_DATASETS
from benchmark.improvement.run import OUT as DEVELOPMENT
from benchmark.preprocessing import segments
from benchmark.real_artifacts import (
    DATASETS,
    complete_by_dataset,
    failures,
    quality_table,
    run_text,
    single_class_shares,
)
from pfc_shared.latex import marked, panel_table, write_table
from pfc_shared.statistics import METRICS, paired_comparisons

OUT = analysis.ROOT.parent.parent / "monografia" / "RealImprovement" / "artifacts"
CONFIRMATION = analysis.ROOT / "results_confirmation"
FINAL_EXPERIMENT = analysis.ROOT / "results_final_model"
TIMING = analysis.ROOT / "results_timing_real"
MEMORY = analysis.ROOT / "results_memory_real"
DEVELOPMENT_RUNS = range(32, 64)
CONFIRMATION_RUNS = range(32)
BENCHMARKS = ["DenStream", "DBSTREAM", "BIRCH", "MiniBatchKMeans", "BICO"]
FINAL = "MicroTEDAClusPlusFinal"

# the steps of the chain, by their names in the improvement experiments and in the chapter
EARLIER = r"\modelname{MicroTEDAClus+ (Ch.~3)}"
CHAIN = {
    "control": EARLIER,
    "low_variance": r"\modelname{low-variance}",
    "no_noise": r"\modelname{low-variance + no-noise}",
    "no_noise_link": r"\modelname{low-variance + no-noise + link-factor}",
}
# the candidates outside the chain, by their names in the improvement experiments and in the
# chapter, with their control, None for the one of their registration; the wide link factor is
# also compared with the last step of the chain, as its round planned
CANDIDATES_OUTSIDE = [
    ("m_const", None, "m-fixed"),
    ("m_scalar", None, "m-constant"),
    ("m_param", None, "m-sigmoid"),
    ("m_reparam", None, "m-reparam"),
    ("m_reparam_up3", None, "m-rate"),
    ("chi_threshold", None, "chi-threshold"),
    ("diagonal", None, "diagonal-variance"),
    ("covariance", None, "covariance"),
    ("volume_density", None, "volume-density"),
    ("log_density", None, "log-density"),
    ("diagonal_log_density", None, "diagonal-variance + log-density"),
    ("no_activation", None, "no-activation"),
    ("mature_mean", None, "mature-mean"),
    ("recency", None, "recency"),
    ("dim_penalty", None, "dimensionality-penalty"),
    ("norm_typicality", None, "norm-typicality"),
    ("local_weights", None, "local-weights"),
    ("link_factor", None, "link-factor"),
    ("no_noise_link_wide", None, "wide-link"),
    ("no_noise_link_wide", "no_noise_link", "wide-link"),
]
NOT_RUN = {"diagonal_log_density"}
OUTCOME = {"+": "$+$", "-": "$-$", "±": r"$\pm$", "=": "$=$"}
CHAIN_METRICS = ("ari_overall", "ari_steadystate", "mc_steadystate")
SINGLE_BATCH_METRICS = ("ari_steadystate",)


def _median(values: pd.Series, metric: str) -> str:
    config = METRICS[metric]
    return config["fmt"].format(
        np.median(config["round"](values.to_numpy(dtype=float)))
    )


def chain_table(steps, failed, datasets, excluded, runs, caption, label, path):
    """Each row against the row above it, over the runs that both finished.

    `steps` maps the label of each step, in order, to its run-level results, and
    `failed` maps it to its number of failed tasks per dataset; `runs` is the
    number of runs of the range.
    """
    labels, results = list(steps), list(steps.values())

    def values(of_step, dataset, metric):
        return of_step[of_step.dataset == dataset].set_index("run")[metric]

    panels = []
    for metric in CHAIN_METRICS:
        shown = [
            d
            for d in datasets
            if not (metric in SINGLE_BATCH_METRICS and d in excluded)
        ]
        rows = []
        for i, (label_of_step, of_step) in enumerate(steps.items()):
            verdicts = {}
            if i > 0:
                pairs = {}
                for d in shown:
                    above, this = values(results[i - 1], d, metric), values(
                        of_step, d, metric
                    )
                    shared = above.index.intersection(this.index)
                    if len(shared) > 1:
                        pairs[d] = (
                            above.loc[shared].to_numpy(),
                            this.loc[shared].to_numpy(),
                        )
                verdicts = paired_comparisons(pairs, metric)["verdict"].to_dict()
            cells = {
                d: marked(
                    _median(values(of_step, d, metric), metric), verdicts.get(d, "same")
                )
                for d in shown
            }
            rows.append((label_of_step, cells, str))
        panels.append((TITLES[metric], rows))
    panels.append(
        (
            "Failed runs",
            [
                (step, {d: failed[step].get(d, 0) for d in datasets}, "{:.0f}".format)
                for step in labels
            ],
        )
    )
    panel_table(
        path,
        f"{caption}: medians over the {runs} runs less the failed ones. Each row is compared "
        "with the row above it over the runs that both finished: a value in bold is "
        "significantly better than that of the row above it, and a value in red "
        "significantly worse; a smaller summary is better. A dash marks the Steady-State ARI "
        "of the streams whose penultimate batch holds a single class. Every failure is a "
        "summary that exceeded a quarter of the tuning segment.",
        label,
        columns(datasets),
        panels,
    )


def development(name: str) -> pd.DataFrame:
    return analysis.run_level(DEVELOPMENT / name)


def failed_tasks(directory, model=None) -> pd.Series:
    """Number of failed tasks per dataset; every failure must be a summary over the limit."""
    tasks = pd.read_csv(directory / "tasks.csv")
    if model is not None:
        tasks = tasks[tasks.model == model]
    failed = tasks[tasks.status != "OK"]
    kinds = set(failed.status.str.extract(r"^ERROR: (\w+)", expand=False))
    assert kinds <= {
        "SummaryLimitExceeded"
    }, f"the captions name only the summary limit: {kinds}"
    return failed.groupby("dataset").size()


def development_chain(excluded):
    chain_table(
        {CHAIN[name]: development(name) for name in CHAIN},
        {CHAIN[name]: failed_tasks(DEVELOPMENT / name) for name in CHAIN},
        DEVELOPMENT_DATASETS,
        excluded,
        len(DEVELOPMENT_RUNS),
        "The kept changes, each applied to the step above it, on the development runs",
        "tab:ri_chain",
        OUT / "chain.tex",
    )


def confirmation_chain(excluded):
    real = analysis.run_level(analysis.REAL)
    steps = {CHAIN["control"]: real[real.model == "MicroTEDAClusPlus"]}
    failed = {CHAIN["control"]: failed_tasks(analysis.REAL, "MicroTEDAClusPlus")}
    for name in ("low_variance", "no_noise", "no_noise_link"):
        steps[CHAIN[name]] = analysis.run_level(CONFIRMATION / name)
        failed[CHAIN[name]] = failed_tasks(CONFIRMATION / name)
    chain_table(
        steps,
        failed,
        DATASETS,
        excluded,
        len(CONFIRMATION_RUNS),
        "The kept changes, each applied to the step above it, on the runs of the "
        "real-data experiment",
        "tab:ri_confirmation",
        OUT / "confirmation.tex",
    )


def variance_ranges() -> pd.DataFrame:
    """Search range of the relative variance by the widened rules in every development
    task, kept in results_improvement/variance_ranges.csv once computed."""
    stored = DEVELOPMENT / "variance_ranges.csv"
    if stored.exists():
        return pd.read_csv(stored)
    rows = []
    for dataset in DEVELOPMENT_DATASETS:
        for run in DEVELOPMENT_RUNS:
            (X_tuning, y_tuning), _ = segments(dataset, run)
            of = widened_ranges_of_categories(X_tuning, y_tuning)[
                Category.RELATIVE_VARIANCE
            ]
            rows.append(
                {"dataset": dataset, "run": run, "low": of.low, "high": of.high}
            )
    ranges = pd.DataFrame(rows)
    ranges.to_csv(stored, index=False)
    return ranges


def _median_qcd(values: pd.Series) -> str:
    q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
    return f"{median:.2f} [{(q3 - q1) / (q3 + q1):.2f}]"


def selection_table():
    """How often the variance limit sits at the lower end of its range, before and after
    its range is extended, and the link factor that the search selects."""
    ranges = variance_ranges()
    searched = {"control": 1, "low_variance": 10}  # the divisor of the lower end
    results = pd.concat([development(name) for name in searched])
    ends = at_the_ends(
        results,
        pd.concat(
            [
                ranges.assign(
                    model=name,
                    hyperparameter="variance_limit",
                    low=ranges.low / divisor,
                    log=True,
                    logit=False,
                )
                for name, divisor in searched.items()
            ]
        ),
    ).set_index(["model", "dataset"])
    percent = "{:.0f}".format
    linked = development("no_noise_link").groupby("dataset")["param:link"]
    panel_table(
        OUT / "selection.tex",
        "Selected hyperparameters along the chain, on the development runs: the percentage "
        "of the runs in which the selected variance limit lies within a twentieth of the "
        "width of its range from its lower end, in the logarithmic scale of the search, "
        "and the median and quartile coefficient of dispersion of the selected link factor.",
        "tab:ri_selection",
        columns(DEVELOPMENT_DATASETS),
        [
            (
                "Variance limit at the lower end of its range",
                [
                    (
                        CHAIN[name],
                        (100 * ends.loc[name, "lower"]).to_dict(),
                        percent,
                    )
                    for name in searched
                ],
            ),
            (
                "Link factor",
                [(CHAIN["no_noise_link"], linked.apply(_median_qcd).to_dict(), str)],
            ),
        ],
    )


def candidates_table():
    lines = [
        r"Candidate & Control & Outcome & Gains & Losses & Lower & Larger & Smaller \\",
        r"\midrule",
    ]
    for name, control, label in CANDIDATES_OUTSIDE:
        control = control or CANDIDATES[name].control
        cells = [rf"\modelname{{{label}}}", CHAIN[control]]
        if name in NOT_RUN:
            cells += ["not run", *["-"] * 5]
        else:
            table = comparison(name, control)
            result, gains, losses = outcome(table)
            overall = table[table.metric == "ari_overall"]
            summary = table[table.metric == "mc_steadystate"].verdict
            cells += [
                OUTCOME[result],
                str(gains),
                str(losses),
                str(int((overall["median"] < overall.control_median).sum())),
                str(int((summary == "worse").sum())),
                str(int((summary == "better").sum())),
            ]
        lines.append(" & ".join(cells) + r" \\")
    write_table(
        OUT / "candidates.tex",
        "The candidates outside the chain of kept changes, each against its control on the "
        "development runs: the outcome of the decision rule, improvement ($+$), regression "
        r"($-$), both ($\pm$) or neither ($=$); its numbers of significant gains and losses "
        "in the Overall ARI of the six datasets and in the Steady-State ARI of the four whose "
        "penultimate batch does not hold a single class; the number of datasets in which its "
        "median Overall ARI is lower than that of its control, significantly or not; and the "
        "numbers of datasets in which its steady-state summary is significantly larger or "
        "smaller than that of its control. The covariance was evaluated on eight runs of three "
        "datasets.",
        "tab:ri_candidates",
        "l l c c c c c c",
        lines,
    )


def diagnosis_table():
    """What MicroTEDAClus+ of Chapter 3 does on the development datasets."""
    replay = pd.read_csv(analysis.REAL / "diagnosis" / "MicroTEDAClusPlus.csv")
    replay = replay[
        (replay.status == "OK")
        & replay.dataset.isin(DEVELOPMENT_DATASETS)
        & replay.run.isin(CONFIRMATION_RUNS)
    ]
    medians = replay.groupby("dataset").median(numeric_only=True)
    runs = replay.groupby("dataset").size().to_dict()
    share, count = "{:.2f}".format, "{:.1f}".format
    rows = [
        ("Share of the samples labelled as noise", "noise", share),
        ("Predicted clusters", "predicted_clusters", count),
        ("Classes", "classes", count),
        ("Microclusters", "microclusters", count),
        ("Mature microclusters", "mature", count),
        ("Active microclusters", "active", count),
    ]
    panel_table(
        OUT / "diagnosis.tex",
        f"{EARLIER} on the real datasets of the development, replayed with the "
        "hyperparameters selected in the real-data experiment: medians over the batches of "
        f"a stream, then medians over {run_text(runs)}. A mature microcluster has three "
        "samples or more.",
        "tab:ri_diagnosis",
        columns(DEVELOPMENT_DATASETS),
        [
            (
                "Per batch",
                [(text, medians[column].to_dict(), fmt) for text, column, fmt in rows],
            )
        ],
        first_column="",
    )


def final_experiment():
    """Results and tasks of the final version and of the benchmark models."""
    real = analysis.run_level(analysis.REAL)
    real_tasks = pd.read_csv(analysis.REAL / "tasks.csv")
    results = pd.concat(
        [real[real.model.isin(BENCHMARKS)], analysis.run_level(FINAL_EXPERIMENT)]
    )
    tasks = pd.concat(
        [
            real_tasks[real_tasks.model.isin(BENCHMARKS)],
            pd.read_csv(FINAL_EXPERIMENT / "tasks.csv"),
        ]
    )
    return results, tasks


def comparison_tables(excluded):
    results, tasks = final_experiment()
    models = [FINAL, *BENCHMARKS]
    complete, runs = complete_by_dataset(results, tasks, models)
    quality_table(
        complete, models, runs, excluded, OUT / "quality.tex", "tab:ri_quality"
    )
    of_final = tasks[tasks.model == FINAL]
    kinds = set(
        of_final.status[of_final.status != "OK"].str.extract(
            r"^ERROR: (\w+)", expand=False
        )
    )
    assert kinds <= {
        "SummaryLimitExceeded"
    }, f"the caption names only the summary limit: {kinds}"
    counts = failures(of_final)
    panel_table(
        OUT / "failures.tex",
        r"Number of runs in which \modelname{MicroTEDAClus+} failed on a real dataset. Every "
        "failure is a summary that exceeded a quarter of the tuning segment.",
        "tab:ri_failures",
        columns(DATASETS),
        [
            (
                "Failed runs",
                [
                    (
                        r"\modelname{MicroTEDAClus+}",
                        counts.loc[FINAL].to_dict(),
                        "{:.0f}".format,
                    )
                ],
            )
        ],
    )
    if (TIMING / "batches").exists() and (MEMORY / "tasks.csv").exists():
        staying = {d: set(complete.model[complete.dataset == d]) for d in DATASETS}
        cost_table(staying, models)


def cost_table(staying, models):
    """Time and memory, per dataset over the runs that every model kept in it finished."""
    timing, memory = analysis.timing_level(TIMING), analysis.memory_level(MEMORY)
    kept_timing, kept_memory, runs = [], [], {}
    for dataset, of_dataset in staying.items():
        t = analysis.on_complete_runs(timing, of_dataset, [dataset])
        m = analysis.on_complete_runs(memory, of_dataset, [dataset])
        shared = sorted(set(t.run) & set(m.run))
        kept_timing.append(t[t.run.isin(shared)])
        kept_memory.append(m[m.run.isin(shared)])
        runs[dataset] = len(shared)
    timing, memory = pd.concat(kept_timing), pd.concat(kept_memory)
    panel_table(
        OUT / "cost.tex",
        "Execution time and memory of the models on the real datasets, each task measured "
        "alone and its memory counted from every allocation of its process: medians over "
        f"{run_text(runs)}. {GROUPS} A dash marks a model left out of a dataset.",
        "tab:ri_cost",
        columns(DATASETS),
        tier_panels(timing, analysis.TIME_METRICS, models, DATASETS, shown=_in_ms)
        + tier_panels(memory, analysis.MEMORY_METRICS, models, DATASETS, shown=_in_kib),
    )


def partitions_table():
    """The partitions that the two versions of MicroTEDAClus+ predict."""
    versions = {
        EARLIER: analysis.REAL / "diagnosis" / "MicroTEDAClusPlus.csv",
        r"\modelname{MicroTEDAClus+}": FINAL_EXPERIMENT / "diagnosis" / f"{FINAL}.csv",
    }
    medians, runs = {}, {}
    for label, path in versions.items():
        replay = pd.read_csv(path)
        replay = replay[(replay.status == "OK") & replay.run.isin(CONFIRMATION_RUNS)]
        medians[label] = replay.groupby("dataset").median(numeric_only=True)
        runs[label] = replay.groupby("dataset").size()
    count, share = "{:.1f}".format, "{:.2f}".format
    first = next(iter(medians.values()))
    panels = [
        (
            "Clusters per batch",
            [("Classes", first["classes"].to_dict(), count)]
            + [
                (label, m["predicted_clusters"].to_dict(), count)
                for label, m in medians.items()
            ],
        ),
        (
            "Share of the samples labelled as noise",
            [(label, m["noise"].to_dict(), share) for label, m in medians.items()],
        ),
        (
            "Homogeneity",
            [
                (label, m["homogeneity"].to_dict(), share)
                for label, m in medians.items()
            ],
        ),
        (
            "Completeness",
            [
                (label, m["completeness"].to_dict(), share)
                for label, m in medians.items()
            ],
        ),
    ]
    fewest = pd.concat(runs.values(), axis=1).min(axis=1).to_dict()
    panel_table(
        OUT / "partitions.tex",
        r"The partitions of the two versions of \MicroTEDAClusPlus{} on the real datasets, replayed "
        "with the hyperparameters selected for each: medians over the batches of a stream, "
        f"then medians over the runs, at least {run_text(fewest)}. Homogeneity and "
        "completeness compare the clusters predicted for a batch with its classes, over the "
        "samples not labelled as noise.",
        "tab:ri_partitions",
        columns(DATASETS),
        panels,
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    shares = single_class_shares(DATASETS, CONFIRMATION_RUNS)
    excluded = set(shares.index[shares["penultimate"] > 0])
    development_shares = single_class_shares(DEVELOPMENT_DATASETS, DEVELOPMENT_RUNS)
    development_excluded = set(
        development_shares.index[development_shares["penultimate"] > 0]
    )
    print(
        "Steady-State ARI left out of", sorted(development_excluded), sorted(excluded)
    )
    development_chain(development_excluded)
    selection_table()
    candidates_table()
    confirmation_chain(excluded)
    if (analysis.REAL / "diagnosis" / "MicroTEDAClusPlus.csv").exists():
        diagnosis_table()
    if (FINAL_EXPERIMENT / "tasks.csv").exists():
        comparison_tables(excluded)
        if (FINAL_EXPERIMENT / "diagnosis" / f"{FINAL}.csv").exists():
            partitions_table()


if __name__ == "__main__":
    main()
