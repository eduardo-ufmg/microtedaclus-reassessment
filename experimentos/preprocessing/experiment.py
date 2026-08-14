"""
Preprocessing-impact experiment.

Investigates how a cumulative chain of offline preprocessing steps affects
MicroTEDAClus on the `cluto-t7-10k` (chameleon ds7) dataset.

Pipeline (cumulative — each stage is derived from the previous one):

    raw -> noise_removed -> downsampled -> scaled -> ordered

Stage definitions (see chat discussion for the rationale):
    raw            : dataset as loaded from the arff file, no operation.
    noise_removed  : drop points whose ground-truth label is -1 (noise class).
    downsampled    : random uniform subsample, 50% of the previous stage's
                      points, original relative order preserved, seed=0.
    scaled         : MinMax scaling to [0, 1] per feature (column).
    ordered        : stable sort by ground-truth label (ascending) — points
                      of the same cluster become contiguous in the stream.

Optimization/evaluation protocol:
    - 20% / 80% opt/eval split
    - Optuna study (TPE, seeded per run), 100 trials, 300s timeout,
      objective = ARI - 1e-3 * seconds
    - Batch size 100, streaming predict-then-fit loop
    - N_RUNS = 20 independent tune+evaluate repetitions per stage;
      per-batch ARI/predict-time/fit-time curves are aggregated across
      runs as median with IQR (Q1-Q3) band; tuned hyperparameters are
      aggregated across runs (median [Q1,Q3] for numeric, mode for
      categorical) and reported as text on each stage's ARI plot.
"""

import os
import time
from io import StringIO

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import urllib.request
from scipy.io import arff
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from MicroTEDAClus.MicroTEDAClus import (
    MicroTEDAClus,
    MicroTEDAClusParameterSearchSpace,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)

OUTPUT_DIR = "./plots/preprocessing_study"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CHAMELEON_DS7_URL = (
    "https://raw.githubusercontent.com/deric/clustering-benchmark/master/"
    "src/main/resources/datasets/artificial/cluto-t7-10k.arff"
)

BATCH_SIZE = 100
OPT_TRIALS = 100
OPT_TIMEOUT = 300  # seconds
SPLIT_FRAC = 0.2
DOWNSAMPLE_FRAC = 0.5
RNG_SEED = 0
N_RUNS = 23

STAGES = ["raw", "noise_removed", "downsampled", "ordered", "scaled"]


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
def get_chameleon(url):
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as response:
        data, meta = arff.loadarff(StringIO(response.read().decode("utf-8")))

    df = pd.DataFrame(data)
    target_col = df.columns[-1]
    y_raw = df[target_col].str.decode("utf-8")
    y_raw = y_raw.replace("noise", "-1")
    y = y_raw.astype(int).values
    X = df.drop(columns=[target_col]).values
    return X, y


# --------------------------------------------------------------------------
# Preprocessing operators (each takes the *previous* stage's (X, y))
# --------------------------------------------------------------------------
def op_noise_removed(X, y):
    mask = y != -1
    return X[mask], y[mask]


def op_downsampled(X, y, frac=DOWNSAMPLE_FRAC, rng=None):
    rng = rng or np.random.default_rng(RNG_SEED)
    n_keep = max(1, int(round(len(X) * frac)))
    idx = rng.choice(len(X), size=n_keep, replace=False)
    idx = np.sort(idx)  # preserve relative order; "ordered" stage handles order
    return X[idx], y[idx]


def op_scaled(X, y):
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y


def op_ordered(X, y):
    idx = np.argsort(y, kind="stable")
    return X[idx], y[idx]


def build_pipeline(X0, y0):
    """Returns dict stage_name -> (X, y), built cumulatively."""
    datasets = {}
    X, y = X0, y0
    datasets["raw"] = (X, y)

    X, y = op_noise_removed(X, y)
    datasets["noise_removed"] = (X, y)

    X, y = op_downsampled(X, y)
    datasets["downsampled"] = (X, y)

    X, y = op_scaled(X, y)
    datasets["scaled"] = (X, y)

    X, y = op_ordered(X, y)
    datasets["ordered"] = (X, y)

    return datasets


# --------------------------------------------------------------------------
# Optuna objective
# --------------------------------------------------------------------------
def objective(trial, X_opt, y_opt, batch_size):
    params = MicroTEDAClusParameterSearchSpace().sample_parameters(trial)

    try:
        model = MicroTEDAClus(**params)
    except Exception as e:
        print(f"Error initializing MicroTEDAClus with params {params}: {e}")
        raise optuna.exceptions.TrialPruned()

    burn_in_idx = int(0.05 * len(X_opt))
    y_pred_all = []
    num_batches = len(X_opt) // batch_size

    t0 = time.process_time_ns()

    try:
        for b in range(num_batches):
            start_idx = b * batch_size
            end_idx = start_idx + batch_size
            X_batch = X_opt[start_idx:end_idx]

            preds = model.predict(X_batch)
            y_pred_all.extend(preds)
            model.fit(X_batch)

    except Exception as e:
        print(f"Error occurred while executing MicroTEDAClus: {e}")
        raise optuna.exceptions.TrialPruned()

    t_total_sec = (time.process_time_ns() - t0) / 1e9

    y_pred_eval = np.array(y_pred_all)[burn_in_idx:]
    y_opt_eval = y_opt[: len(y_pred_all)][burn_in_idx:]

    ari = adjusted_rand_score(y_opt_eval, y_pred_eval)

    l2_time_penalty = 1e-3 * t_total_sec
    return ari - l2_time_penalty


# --------------------------------------------------------------------------
# Evaluation loop (single run; tracks only ari, predict/fit time)
# --------------------------------------------------------------------------
def evaluate(model, X_eval, y_eval, batch_size):
    ari_list = []
    t_predict_ms = []
    t_fit_ms = []

    num_batches = len(X_eval) // batch_size

    for b in range(num_batches):
        start_idx = b * batch_size
        end_idx = start_idx + batch_size
        X_batch = X_eval[start_idx:end_idx]
        y_batch = y_eval[start_idx:end_idx]

        try:
            t0_pred = time.process_time_ns()
            preds = model.predict(X_batch)
            t_pred_ns = time.process_time_ns() - t0_pred

            t0_fit = time.process_time_ns()
            model.fit(X_batch)
            t_fit_ns = time.process_time_ns() - t0_fit

            ari = adjusted_rand_score(np.asarray(y_batch), preds)

        except Exception as e:
            print(f"Error at batch {b}: {e}")
            ari, t_pred_ns, t_fit_ns = 0.0, 0, 0

        ari_list.append(ari)
        t_predict_ms.append(t_pred_ns / 1e6)
        t_fit_ms.append(t_fit_ns / 1e6)

    return np.array(ari_list), np.array(t_predict_ms), np.array(t_fit_ms)


# --------------------------------------------------------------------------
# Repeated runs (tuning + evaluation) and cross-run aggregation
# --------------------------------------------------------------------------
def tune_and_evaluate_once(
    stage, run_idx, X_opt, y_opt, X_eval, y_eval, batch_size, n_eval_batches
):
    """One independent tuning+evaluation repetition. Returns:
    (best_params: dict, ari: (n_eval_batches,), t_predict_ms: (...,), t_fit_ms: (...,))
    On an all-pruned study, best_params is {} and the arrays are NaN-filled
    (so they still align for cross-run aggregation, but don't bias it).
    """
    seed = RNG_SEED + 1000 * STAGES.index(stage) + run_idx
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(
        lambda trial: objective(trial, X_opt, y_opt, batch_size),
        n_trials=OPT_TRIALS,
        timeout=OPT_TIMEOUT,
    )

    try:
        best_params = study.best_params
        model = MicroTEDAClus(**best_params)
    except ValueError:
        nan_arr = np.full(n_eval_batches, np.nan)
        return {}, nan_arr, nan_arr.copy(), nan_arr.copy()

    ari, t_predict_ms, t_fit_ms = evaluate(model, X_eval, y_eval, batch_size)
    return best_params, ari, t_predict_ms, t_fit_ms


def aggregate_curves(arr_list):
    """arr_list: list of 1D arrays of equal length (may contain NaNs for
    failed runs). Returns (median, q1, q3) along the run axis, ignoring
    NaNs. Median is used as the central statistic; Q1-Q3 provide the
    variability measure representing the interquartile range (IQR)."""
    stacked = np.vstack(arr_list)
    with np.errstate(all="ignore"):
        median = np.nanmedian(stacked, axis=0)
        q1 = np.nanpercentile(stacked, 25, axis=0)
        q3 = np.nanpercentile(stacked, 75, axis=0)
    return median, q1, q3


def aggregate_params(params_runs):
    """params_runs: list of dicts, one per run ({} for an all-pruned run).
    Numeric parameters are summarized by median [Q1, Q3] across valid runs;
    non-numeric (categorical) parameters by their mode. Returns
    (summary_dict[name -> formatted str], n_valid_runs)."""
    valid = [p for p in params_runs if p]
    if not valid:
        return {}, 0

    keys = valid[0].keys()
    summary = {}
    for k in keys:
        vals = [p[k] for p in valid if k in p]
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals):
            med = np.median(vals)
            q1 = np.percentile(vals, 25)
            q3 = np.percentile(vals, 75)

            if all(v > 1 for v in vals):
                summary[k] = f"{int(med)} [{int(q1)}, {int(q3)}]"
            else:
                summary[k] = f"{med:.1g} [{q1:.1g}, {q3:.1g}]"
        else:
            from collections import Counter

            top, cnt = Counter(vals).most_common(1)[0]
            summary[k] = f"{top} ({cnt}/{len(vals)} runs)"
    return summary, len(valid)


def format_params_text(param_summary, n_valid, n_runs):
    if n_valid == 0:
        return "no valid trial in any run (all pruned)"
    lines = [f"{k} = {v}" for k, v in param_summary.items()]
    if n_valid < n_runs:
        lines.append(f"[{n_valid}/{n_runs} runs valid]")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------
def plot_dataset(X, y, stage, out_dir):
    fig, ax = plt.subplots()
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y)
    ax.set_title(f"{stage} (n={len(X)})")
    ax.set_xlabel("$x_0$")
    ax.set_ylabel("$x_1$")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{stage}_dataset.png"))
    plt.close(fig)


def plot_metrics(ari_stats, t_predict_stats, t_fit_stats, params_text, stage, out_dir):
    """ari_stats, t_predict_stats, t_fit_stats: (median, q1, q3) tuples,
    each aggregated across N_RUNS independent tuning+evaluation repetitions
    (see tune_and_evaluate_once / aggregate_curves)."""
    ari_med, ari_q1, ari_q3 = ari_stats
    tp_med, tp_q1, tp_q3 = t_predict_stats
    tf_med, tf_q1, tf_q3 = t_fit_stats
    batches = np.arange(len(ari_med))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(batches, ari_med, color="tab:blue", linewidth=2.0, label="median")
    axes[0].fill_between(
        batches, ari_q1, ari_q3, color="tab:blue", alpha=0.3, label="IQR"
    )
    axes[0].set_title(f"{stage} — ARI per Batch")
    axes[0].set_xlabel("Batch Index")
    axes[0].set_ylabel("Adjusted Rand Index")
    axes[0].set_ylim(-1.0, 1.0)
    axes[0].legend(fontsize=7, loc="upper right")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    axes[0].text(
        0.02,
        0.03,
        params_text,
        transform=axes[0].transAxes,
        ha="left",
        va="bottom",
        bbox=dict(
            boxstyle="round,pad=0.35", facecolor="white", edgecolor="gray", alpha=0.85
        ),
    )

    axes[1].plot(batches, tp_med, label="predict", color="tab:orange", linewidth=2.0)
    axes[1].fill_between(
        batches, tp_q1, tp_q3, color="tab:orange", alpha=0.3, label="predict IQR"
    )
    axes[1].plot(batches, tf_med, label="fit", color="tab:green", linewidth=2.0)
    axes[1].fill_between(
        batches, tf_q1, tf_q3, color="tab:green", alpha=0.3, label="fit IQR"
    )
    axes[1].set_title(f"{stage} — Processing Time per Batch")
    axes[1].set_xlabel("Batch Index")
    axes[1].set_ylabel("Time (ms)")
    axes[1].set_yscale("log")
    axes[1].legend(fontsize=7)
    axes[1].grid(True, which="both", linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{stage}_metrics.png"))
    plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    print("Loading chameleon ds7 (cluto-t7-10k)...")
    X0, y0 = get_chameleon(CHAMELEON_DS7_URL)

    print("Building cumulative preprocessing pipeline...")
    datasets = build_pipeline(X0, y0)

    for stage in tqdm(STAGES, desc="Stages"):
        X, y = datasets[stage]

        plot_dataset(X, y, stage, OUTPUT_DIR)

        n_samples = len(X)
        split_idx = int(SPLIT_FRAC * n_samples)
        X_opt, y_opt = X[:split_idx], y[:split_idx]
        X_eval, y_eval = X[split_idx:], y[split_idx:]
        n_eval_batches = len(X_eval) // BATCH_SIZE

        params_runs = []
        ari_runs, tp_runs, tf_runs = [], [], []

        for run_idx in tqdm(range(N_RUNS), desc=f"{stage} runs", leave=False):
            params, ari, t_predict_ms, t_fit_ms = tune_and_evaluate_once(
                stage,
                run_idx,
                X_opt,
                y_opt,
                X_eval,
                y_eval,
                BATCH_SIZE,
                n_eval_batches,
            )
            params_runs.append(params)
            ari_runs.append(ari)
            tp_runs.append(t_predict_ms)
            tf_runs.append(t_fit_ms)

        param_summary, n_valid = aggregate_params(params_runs)
        params_text = format_params_text(param_summary, n_valid, N_RUNS)

        if n_valid == 0:
            print(f"[{stage}] All trials pruned in every run — skipping plot.")
            continue

        ari_stats = aggregate_curves(ari_runs)
        t_predict_stats = aggregate_curves(tp_runs)
        t_fit_stats = aggregate_curves(tf_runs)

        plot_metrics(
            ari_stats, t_predict_stats, t_fit_stats, params_text, stage, OUTPUT_DIR
        )

    print(f"Done. Plots written to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
