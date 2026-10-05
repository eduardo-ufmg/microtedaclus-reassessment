"""Structural Replay Harness for the Stage 5 Macrocluster Topology Assessment.

Replays MicroTEDAClusCorrectedTypi (Stage 3 Control) with its officially tuned
parameters on the same evaluation streams used by experiment_runner.py and, at
every batch, labels the same model state under both topologies:

1. Two-pass active graph G1 (retained implementation behaviour)
2. Single intersection graph G0 with activation applied only when weighting

Because both labellings are computed from one model state, every difference is
attributable to the topology alone. The harness also records the structural
composition of the microcluster population and the share of the fitting time
spent in the activation pass, and emits a LaTeX table in the chapter's style.
"""

import argparse
import os
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.metrics import adjusted_rand_score

# Set MPLCONFIGDIR to avoid warnings
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-topology-replay")

# Ensure repository root is on path
CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from editorial_config import ITERATIVE_DATASET_ORDER, dataset_display_name
from main_experiment import get_dataset
from models.MicroTEDAClusCorrected.MicroTEDAClusCorrected_typi import (
    MicroTEDAClusCorrectedTypi,
)
from stages_config import DEFAULT_RUNS, STAGE_MACROCLUSTER_ACTIVATION

CONTROL_TAG = "MicroTEDAClusCorrected_typi"
BATCH_SIZE = 100
SPLIT_FRACTION = 0.2
STEADY_STATE_FRACTION = 0.8
BASE_SEED = 0

DIR_PARAMS_EXP = REPO_ROOT / "results_params_experiments"
DIR_REPLAY_RESULTS = CURRENT_DIR / "results"
DIR_TABLES = (
    REPO_ROOT.parent.parent
    / "monografia"
    / STAGE_MACROCLUSTER_ACTIVATION.chapter
    / STAGE_MACROCLUSTER_ACTIVATION.artifact_dir
    / "tables"
)

LOCAL_ARFF = {
    "chameleon_ds3": "cluto-t7-10k.arff",
    "chameleon_ds4": "cluto-t8-8k.arff",
}

# (column, header, format)
TABLE_COLUMNS = [
    ("immature", "Immature", "{:.0f}"),
    ("active", "Active", "{:.0f}"),
    ("g1_components", r"$\mathbb{G}_1$ Components", "{:.0f}"),
    ("g0_active_components", r"Active $\mathbb{G}_0$ Components", "{:.0f}"),
    ("unassigned_pct", r"Unassigned [\%]", "{:.1f}"),
    ("agreement", "Labelling Agreement", "{:.3f}"),
    ("ari_g1", r"ARI under $\mathbb{G}_1$", "{:.3f}"),
    ("ari_g0", r"ARI under $\mathbb{G}_0$", "{:.3f}"),
    ("activation_time_pct", r"Activation Pass [\% of Fit]", "{:.2f}"),
]


def load_dataset(dataset_name: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Load dataset, using local cached arff files when available."""
    if dataset_name in LOCAL_ARFF:
        local_path = REPO_ROOT / "data_samples" / LOCAL_ARFF[dataset_name]
        if local_path.exists():
            data, _ = arff.loadarff(str(local_path))
            df = pd.DataFrame(data)
            target_col = df.columns[-1]
            y_raw = df[target_col].str.decode("utf-8").replace("noise", "-1")
            X = df.drop(columns=[target_col]).to_numpy()
            y = y_raw.astype(int).to_numpy()
            return X, y
    return get_dataset(dataset_name, seed)


def load_control_params(dataset: str, run: int) -> dict | None:
    """Retrieve the officially tuned parameters of the control model."""
    param_file = DIR_PARAMS_EXP / f"{dataset}_{CONTROL_TAG}_{run}.csv"
    if not param_file.exists():
        return None
    df_p = pd.read_csv(param_file)
    if df_p.empty:
        return None
    params = dict(zip(df_p["param_name"], df_p["param_value"]))
    if "decay" in params:
        params["decay"] = int(params["decay"])
    return params


def predict_single_graph(model, X: np.ndarray) -> tuple[np.ndarray, int]:
    """Label samples from the components of G0, weighting only active microclusters."""
    by_id = {m["id"]: m for m in model.micro_clusters}
    groups = []
    for mg in model.macro_clusters:
        active = [by_id[i] for i in mg if i in by_id and by_id[i]["active"]]
        if active:
            groups.append(active)

    labels = np.full(len(X), -1, dtype=int)
    if groups:
        for index, xk in enumerate(X):
            memberships = [
                MicroTEDAClusCorrectedTypi.calculate_membership(xk, g) for g in groups
            ]
            labels[index] = int(np.argmax(memberships))
    return labels, len(groups)


def replay(dataset: str, run: int, params: dict) -> pd.DataFrame:
    """Replay one (dataset, run) task mirroring experiment_runner.py."""
    dataset_hash = zlib.crc32(dataset.encode("utf-8"))
    run_seed = BASE_SEED + run + (dataset_hash % 10000)
    X, y = load_dataset(dataset, run_seed)
    split_idx = int(SPLIT_FRACTION * len(X))
    X_eval, y_eval = X[split_idx:], y[split_idx:]

    model = MicroTEDAClusCorrectedTypi(**params)

    activation_ns = [0]
    original_define_activations = model.define_activations

    def timed_define_activations():
        t0 = time.process_time_ns()
        original_define_activations()
        activation_ns[0] += time.process_time_ns() - t0

    model.define_activations = timed_define_activations

    records = []
    for b in range(len(X_eval) // BATCH_SIZE):
        start_idx, end_idx = b * BATCH_SIZE, (b + 1) * BATCH_SIZE
        X_batch, y_batch = X_eval[start_idx:end_idx], y_eval[start_idx:end_idx]

        preds_g1 = model.predict(X_batch).copy()
        preds_g0, g0_active_components = predict_single_graph(model, X_batch)

        mcs = model.micro_clusters
        records.append(
            {
                "dataset": dataset,
                "run": run,
                "batch": b,
                "ari_g1": adjusted_rand_score(y_batch, preds_g1),
                "ari_g0": adjusted_rand_score(y_batch, preds_g0),
                "agreement": adjusted_rand_score(preds_g1, preds_g0),
                "mc": len(mcs),
                "immature": sum(m["num_samples"] < 3 for m in mcs),
                "active": sum(m["active"] and m["num_samples"] >= 3 for m in mcs),
                "g1_components": len(model.active_macro_clusters),
                "g0_components": len(model.macro_clusters),
                "g0_active_components": g0_active_components,
                "unassigned_pct": 100.0 * float(np.mean(preds_g1 == -1)),
            }
        )

        activation_ns[0] = 0
        t0_fit = time.process_time_ns()
        model.fit(X_batch)
        records[-1]["time_fit_cpu_ns"] = time.process_time_ns() - t0_fit
        records[-1]["time_activation_cpu_ns"] = activation_ns[0]

    return pd.DataFrame(records)


def summarize_run(df: pd.DataFrame) -> dict:
    """Reduce a batch trajectory to run-level scalars (steady-state window as in analysis_pipeline.py)."""
    ss = df.iloc[int(len(df) * STEADY_STATE_FRACTION) :]
    fit_ns = df["time_fit_cpu_ns"].sum()
    return {
        "dataset": df["dataset"].iloc[0],
        "run": df["run"].iloc[0],
        "mc": ss["mc"].mean(),
        "immature": ss["immature"].mean(),
        "active": ss["active"].mean(),
        "g1_components": ss["g1_components"].mean(),
        "g0_active_components": ss["g0_active_components"].mean(),
        "unassigned_pct": ss["unassigned_pct"].mean(),
        "agreement": df["agreement"].mean(),
        "ari_g1": df["ari_g1"].mean(),
        "ari_g0": df["ari_g0"].mean(),
        "activation_time_pct": (
            100.0 * df["time_activation_cpu_ns"].sum() / fit_ns if fit_ns > 0 else 0.0
        ),
    }


def generate_latex_table(df_runs: pd.DataFrame, datasets: list[str]):
    """Emit the structural table as median (Q1, Q3) across runs."""
    label_prefix = STAGE_MACROCLUSTER_ACTIVATION.label_prefix
    latex_lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Structural analysis of the \\modelname{corr-typicality} microcluster population under both macrocluster topologies.}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{l " + "c " * len(TABLE_COLUMNS) + "}",
        "\\toprule",
        "Dataset & " + " & ".join(header for _, header, _ in TABLE_COLUMNS) + " \\\\",
        "\\midrule",
    ]

    for dataset in datasets:
        df_ds = df_runs[df_runs["dataset"] == dataset]
        display_name = dataset_display_name(dataset).replace("_", "\\_")
        row_str = f"\\datasetname{{{display_name}}}"
        for column, _, fmt in TABLE_COLUMNS:
            if df_ds.empty:
                row_str += " & -"
                continue
            s = df_ds[column]
            row_str += f" & {fmt.format(s.median())} ({fmt.format(s.quantile(0.25))}, {fmt.format(s.quantile(0.75))})"
        latex_lines.append(row_str + " \\\\")

    latex_lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "}",
            f"\\label{{tab:{label_prefix}structure}}",
            "\\end{table}",
        ]
    )

    DIR_TABLES.mkdir(parents=True, exist_ok=True)
    with open(DIR_TABLES / "structure.tex", "w") as f:
        f.write("\n".join(latex_lines))


def main():
    parser = argparse.ArgumentParser(
        description="Structural replay of the control model under both macrocluster topologies"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(ITERATIVE_DATASET_ORDER),
        help="Datasets to replay (default: all iterative datasets)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=DEFAULT_RUNS,
        help=f"Number of randomized repetitions (default: {DEFAULT_RUNS})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replay tasks even if a cached trajectory exists",
    )
    args = parser.parse_args()

    DIR_REPLAY_RESULTS.mkdir(exist_ok=True)

    run_records = []
    for dataset in args.datasets:
        for run in range(args.runs):
            cache_file = DIR_REPLAY_RESULTS / f"{dataset}_{run}.csv"
            if cache_file.exists() and not args.force:
                df = pd.read_csv(cache_file)
            else:
                params = load_control_params(dataset, run)
                if params is None:
                    print(
                        f"Notice: no tuned {CONTROL_TAG} parameters for {dataset} run {run}. Skipping."
                    )
                    continue
                df = replay(dataset, run, params)
                df.to_csv(cache_file, index=False)

            if df.empty:
                continue
            run_records.append(summarize_run(df))
            print(
                f"{dataset:20s} run {run:2d} | ARI G1 {run_records[-1]['ari_g1']:.3f} | ARI G0 {run_records[-1]['ari_g0']:.3f} | agreement {run_records[-1]['agreement']:.3f}",
                flush=True,
            )

    if not run_records:
        print("No tasks replayed.")
        return

    df_runs = pd.DataFrame(run_records)
    df_runs.to_csv(DIR_REPLAY_RESULTS / "run_level.csv", index=False)
    generate_latex_table(df_runs, args.datasets)

    pd.set_option("display.width", 200)
    print("\nMedians across runs:")
    print(df_runs.drop(columns=["run"]).groupby("dataset").median().round(3))
    print(f"\nStructural table written to {DIR_TABLES / 'structure.tex'}")


if __name__ == "__main__":
    main()
