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

OUTPUT_DIR = "./plots/scaling_study"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CHAMELEON_DS7_URL = (
    "https://raw.githubusercontent.com/deric/clustering-benchmark/master/"
    "src/main/resources/datasets/artificial/cluto-t7-10k.arff"
)

BATCH_SIZE = 100
OPT_TRIALS = 100
OPT_TIMEOUT = 600  # seconds
SPLIT_FRAC = 0.2
DOWNSAMPLE_FRAC = 0.5
RNG_SEED = 0
N_RUNS = 100

CONDITIONS = ["minmax", "mean", "global"]


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
# Preprocessing operators
# --------------------------------------------------------------------------
def op_noise_removed(X, y):
    mask = y != -1
    return X[mask], y[mask]


def op_downsampled(X, y, frac=DOWNSAMPLE_FRAC, rng=None):
    rng = rng or np.random.default_rng(RNG_SEED)
    n_keep = max(1, int(round(len(X) * frac)))
    idx = rng.choice(len(X), size=n_keep, replace=False)
    idx = np.sort(idx)  # preserve relative order
    return X[idx], y[idx]


def build_base(X0, y0):
    """raw -> noise_removed -> downsampled."""
    X, y = op_noise_removed(X0, y0)
    X, y = op_downsampled(X, y)
    return X, y


def build_conditions(X_base, y_base):
    """Returns dict condition_name -> (X, y), all derived from X_base."""
    assert X_base.shape[1] == 2, (
        f"Expected a 2D dataset (x0, x1), got {X_base.shape[1]} features. "
        "The constants defined below (mean/max over x0, x1) assume 2D."
    )

    max_x0 = X_base[:, 0].max()
    max_x1 = X_base[:, 1].max()
    c_scaled_back = np.mean([max_x0, max_x1])
    g_global = np.max([max_x0, max_x1])

    datasets = {}

    scaler = MinMaxScaler()
    X_minmax = scaler.fit_transform(X_base)
    datasets["minmax"] = (X_minmax, y_base)

    X_scaled_back = X_minmax * c_scaled_back
    datasets["mean"] = (X_scaled_back, y_base)

    X_global = X_base / g_global
    datasets["global"] = (X_global, y_base)

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
        print(f"Error occurred while fitting MicroTEDAClus: {e}")
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
    condition, run_idx, X_opt, y_opt, X_eval, y_eval, batch_size, n_eval_batches
):
    """One independent tuning+evaluation repetition. Returns:
    (best_params: dict, ari: (n_eval_batches,), t_predict_ms: (...,), t_fit_ms: (...,))
    On an all-pruned study, best_params is {} and the arrays are NaN-filled
    (so they still align for cross-run aggregation, but don't bias it).
    """
    seed = RNG_SEED + 1000 * CONDITIONS.index(condition) + run_idx
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


def plot_grid(datasets, results, out_dir):
    """
    datasets: list of datasets (X, y) for each condition
    results: dict condition_name -> {
        'ari': (median, q1, q3), 't_predict': (median, q1, q3),
        't_fit': (median, q1, q3), 'params_text': str
    }, each median/q1/q3 computed across N_RUNS independent tuning+
    evaluation repetitions (see tune_and_evaluate_once / aggregate_curves).
    Produces a single len(CONDITIONS) x 3 figure:
        rows    = conditions (in CONDITIONS order)
        columns = [dataset scatter, ARI per batch, processing time per batch]
    with a shared, fixed y range per column, except for the scatter plots.
    Median is drawn as a solid line, with a shaded region representing the
    interquartile range (IQR, Q1-Q3). The lower portion of each ARI
    subplot (near the -1 floor, effectively unused by the ARI series post
    burn-in) is used to display the aggregated tuned hyperparameters.
    """
    n_rows = len(CONDITIONS)

    # Fixed y range, column 0 (ARI): ARI is bounded in [-1, 1] by definition.
    ari_ylim = (-1.0, 1.0)

    # Fixed y range, column 1 (time, log scale): derived from the actual
    # data across all conditions (using the Q1/Q3 bounds).
    all_times = np.concatenate(
        [
            np.concatenate(
                [
                    results[c]["t_predict"][1],
                    results[c]["t_predict"][2],
                    results[c]["t_fit"][1],
                    results[c]["t_fit"][2],
                ]
            )
            for c in CONDITIONS
            if len(results[c]["t_predict"][0]) > 0
        ]
    )
    all_times = all_times[~np.isnan(all_times)]
    all_times_pos = all_times[all_times > 0]
    if len(all_times_pos) > 0:
        time_ylim = (all_times_pos.min() * 0.8, all_times.max() * 1.25)
    else:
        time_ylim = (1e-3, 1.0)

    fig, axes = plt.subplots(n_rows, 3, figsize=(14, 4.5 * n_rows), squeeze=False)

    for row, condition in enumerate(CONDITIONS):
        ari_med, ari_q1, ari_q3 = results[condition]["ari"]
        tp_med, tp_q1, tp_q3 = results[condition]["t_predict"]
        tf_med, tf_q1, tf_q3 = results[condition]["t_fit"]
        batches = np.arange(len(ari_med))

        ax_scatter = axes[row, 0]
        ax_ari = axes[row, 1]
        ax_time = axes[row, 2]

        X, y = datasets[condition]
        ax_scatter.scatter(X[:, 0], X[:, 1], c=y)
        ax_scatter.set_title(f"{condition} — Dataset")
        ax_scatter.set_xlabel("$x_0$")
        ax_scatter.set_ylabel("$x_1$")

        ax_ari.plot(batches, ari_med, color="tab:blue", linewidth=2.0, label="median")
        ax_ari.fill_between(
            batches, ari_q1, ari_q3, color="tab:blue", alpha=0.3, label="IQR"
        )
        ax_ari.set_title(f"{condition} — ARI per batch")
        ax_ari.set_xlabel("Batch Index")
        ax_ari.set_ylabel("Adjusted Rand Index")
        ax_ari.set_ylim(*ari_ylim)
        ax_ari.legend(fontsize=7, loc="upper right")
        ax_ari.grid(True, linestyle="--", alpha=0.5)

        ax_ari.text(
            0.02,
            0.03,
            results[condition]["params_text"],
            transform=ax_ari.transAxes,
            ha="left",
            va="bottom",
            bbox=dict(
                boxstyle="round,pad=0.35",
                facecolor="white",
                edgecolor="gray",
                alpha=0.85,
            ),
        )

        ax_time.plot(
            batches, tp_med, label="predict", color="tab:orange", linewidth=2.0
        )
        ax_time.fill_between(
            batches, tp_q1, tp_q3, color="tab:orange", alpha=0.3, label="predict IQR"
        )

        ax_time.plot(batches, tf_med, label="fit", color="tab:green", linewidth=2.0)
        ax_time.fill_between(
            batches, tf_q1, tf_q3, color="tab:green", alpha=0.3, label="fit IQR"
        )

        ax_time.set_title(f"{condition} — processing time per batch")
        ax_time.set_xlabel("Batch Index")
        ax_time.set_ylabel("Time (ms)")
        ax_time.set_yscale("log")
        ax_time.set_ylim(*time_ylim)
        ax_time.legend(fontsize=7)
        ax_time.grid(True, which="both", linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "scaling_study_grid.png"))
    plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    print("Loading chameleon ds7 (cluto-t7-10k)...")
    X0, y0 = get_chameleon(CHAMELEON_DS7_URL)

    print("Building base dataset (raw -> noise_removed -> downsampled)...")
    X_base, y_base = build_base(X0, y0)

    print("Deriving scaling conditions from the base dataset...")
    datasets = build_conditions(X_base, y_base)

    results = {}

    for condition in tqdm(CONDITIONS, desc="Conditions"):
        X, y = datasets[condition]

        n_samples = len(X)
        split_idx = int(SPLIT_FRAC * n_samples)
        X_opt, y_opt = X[:split_idx], y[:split_idx]
        X_eval, y_eval = X[split_idx:], y[split_idx:]
        n_eval_batches = len(X_eval) // BATCH_SIZE

        params_runs = []
        ari_runs, tp_runs, tf_runs = [], [], []

        for run_idx in tqdm(range(N_RUNS), desc=f"{condition} runs", leave=False):
            params, ari, t_predict_ms, t_fit_ms = tune_and_evaluate_once(
                condition,
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

        results[condition] = {
            "ari": aggregate_curves(ari_runs),
            "t_predict": aggregate_curves(tp_runs),
            "t_fit": aggregate_curves(tf_runs),
            "params_text": format_params_text(param_summary, n_valid, N_RUNS),
        }

    print("Plotting combined 3x3 grid...")
    plot_grid(datasets, results, OUTPUT_DIR)

    print(f"Done. Plot written to {OUTPUT_DIR}/scaling_study_grid.png")


if __name__ == "__main__":
    main()
