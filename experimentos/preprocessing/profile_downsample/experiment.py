import os
import time
from collections import defaultdict
from io import StringIO

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import urllib.request
from scipy.io import arff
from sklearn.metrics import adjusted_rand_score
from tqdm import tqdm

from MicroTEDAClus.MicroTEDAClus import (
    MicroTEDAClus,
    MicroTEDAClusParameterSearchSpace,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)

OUTPUT_DIR = "./plots/instrumentation_study"
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
N_RUNS = 20

STAGES = ["noise_removed", "downsampled"]

TIMER_KEYS = [
    "update_micro_clusters",
    "prune_micro_clusters",
    "define_macro_clusters",
    "define_activations",
    "get_active_micro_clusters",
    "calculate_membership",
]


# -----------------------------
# Data loading / preprocessing
# -----------------------------
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


def op_noise_removed(X, y):
    mask = y != -1
    return X[mask], y[mask]


def op_downsampled(X, y, frac=DOWNSAMPLE_FRAC, rng=None):
    rng = rng or np.random.default_rng(RNG_SEED)
    n_keep = max(1, int(round(len(X) * frac)))
    idx = rng.choice(len(X), size=n_keep, replace=False)
    idx = np.sort(idx)
    return X[idx], y[idx]


# -------------------
# Instrumented model
# -------------------
class InstrumentedMicroTEDAClus(MicroTEDAClus):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._timers = defaultdict(float)
        self._births = 0

    def reset_batch_stats(self):
        self._timers = defaultdict(float)
        self._births = 0

    def get_timers_ms(self):
        return {k: self._timers.get(k, 0.0) / 1e6 for k in TIMER_KEYS}

    def create_new_micro_cluster(self, x):
        self._births += 1
        return super().create_new_micro_cluster(x)

    def num_samples_stats(self):
        if not self.micro_clusters:
            return 0.0, 0.0
        counts = [m["num_samples"] for m in self.micro_clusters]
        return float(np.mean(counts)), float(np.median(counts))

    def update_micro_clusters(self, xk):
        t0 = time.process_time_ns()
        result = super().update_micro_clusters(xk)
        self._timers["update_micro_clusters"] += time.process_time_ns() - t0
        return result

    def prune_micro_clusters(self):
        t0 = time.process_time_ns()
        result = super().prune_micro_clusters()
        self._timers["prune_micro_clusters"] += time.process_time_ns() - t0
        return result

    def define_macro_clusters(self):
        t0 = time.process_time_ns()
        result = super().define_macro_clusters()
        self._timers["define_macro_clusters"] += time.process_time_ns() - t0
        return result

    def define_activations(self):
        t0 = time.process_time_ns()
        result = super().define_activations()
        self._timers["define_activations"] += time.process_time_ns() - t0
        return result

    def get_active_micro_clusters(self, mg):
        t0 = time.process_time_ns()
        result = super().get_active_micro_clusters(mg)
        self._timers["get_active_micro_clusters"] += time.process_time_ns() - t0
        return result

    def predict(self, X):
        self.labels_ = np.zeros(len(X), dtype=int)

        for index, xk in enumerate(X):
            memberships = []
            for mg in self.active_macro_clusters:
                active_micro_clusters = self.get_active_micro_clusters(mg)

                t0 = time.process_time_ns()
                membership = MicroTEDAClus.calculate_membership(
                    xk, active_micro_clusters
                )
                self._timers["calculate_membership"] += time.process_time_ns() - t0

                memberships.append(membership)

            # Redundant check corrected
            if len(memberships) > 0:
                self.labels_[index] = np.argmax(memberships)
            else:
                self.labels_[index] = -1

        return self.labels_


# -------------------------------
# Optuna objective — plain model
# -------------------------------
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


# -----------------------------
# Instrumented evaluation loop
# -----------------------------
def evaluate_instrumented(model, X_eval, y_eval, batch_size):
    num_batches = len(X_eval) // batch_size
    records = []

    for b in range(num_batches):
        start_idx = b * batch_size
        end_idx = start_idx + batch_size
        X_batch = X_eval[start_idx:end_idx]

        model.reset_batch_stats()
        try:
            model.predict(X_batch)
            model.fit(X_batch)
        except Exception as e:
            print(f"Error at batch {b}: {e}")

        mean_ns, median_ns = model.num_samples_stats()
        record = {
            "batch": b,
            "cumulative_samples": end_idx,
            "microcluster_count": model.microcluster_count(),
            "births": model._births,
            "mean_num_samples": mean_ns,
            "median_num_samples": median_ns,
        }
        record.update(model.get_timers_ms())
        records.append(record)

    return pd.DataFrame(records)


def aggregate_curves(arr_list):
    """arr_list: list of 1D arrays of equal length (may contain NaNs for
    failed runs). Returns (median, q1, q3) along the run axis, ignoring
    NaNs. Median is the central statistic; Q1-Q3 give the IQR band."""
    stacked = np.vstack(arr_list)
    with np.errstate(all="ignore"):
        median = np.nanmedian(stacked, axis=0)
        q1 = np.nanpercentile(stacked, 25, axis=0)
        q3 = np.nanpercentile(stacked, 75, axis=0)
    return median, q1, q3


# ---------
# Plotting
# ---------
def plot_stage_comparisons(results, out_dir):
    """results: dict stage_name -> {
        'x': cumulative_samples array,
        'microcluster_count': (median, q1, q3),
        'total_time': (median, q1, q3),
    }, each median/q1/q3 aggregated across N_RUNS independent tuning+
    evaluation repetitions (see run_stage / aggregate_curves)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    colors = {"noise_removed": "tab:blue", "downsampled": "tab:red"}

    for stage_name, res in results.items():
        color = colors.get(stage_name)
        x_vals = res["x"]

        # Subplot 1: Microclusters
        mc_med, mc_q1, mc_q3 = res["microcluster_count"]
        ax1.plot(x_vals, mc_med, label=stage_name, color=color, linewidth=2.0)
        ax1.fill_between(x_vals, mc_q1, mc_q3, color=color, alpha=0.25)

        # Subplot 2: Total Timings
        tt_med, tt_q1, tt_q3 = res["total_time"]
        ax2.plot(x_vals, tt_med, label=stage_name, color=color, linewidth=2.0)
        ax2.fill_between(x_vals, tt_q1, tt_q3, color=color, alpha=0.25)

    # Configure Ax 1
    ax1.set_title("Microcluster Count vs. Samples Processed")
    ax1.set_xlabel("Cumulative Samples Processed")
    ax1.set_ylabel("Microcluster Count")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Configure Ax 2
    ax2.set_title("Processing Time vs. Samples Processed")
    ax2.set_xlabel("Cumulative Samples Processed")
    ax2.set_ylabel("Time (ms)")
    ax2.set_yscale("log")
    ax2.legend()
    ax2.grid(True, which="both", linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "comparison_metrics.png"))
    plt.close(fig)


# -----
# Main
# -----
def tune_and_evaluate_once(
    stage_name, run_idx, X_opt, y_opt, X_eval, y_eval, batch_size
):
    """One independent tuning+evaluation repetition. Returns
    (microcluster_count: (n_eval_batches,), total_time_ms: (...,)), NaN-filled
    if the study produced no valid trial (so runs still align for
    cross-run aggregation without biasing it)."""
    n_eval_batches = len(X_eval) // batch_size

    seed = RNG_SEED + 1000 * STAGES.index(stage_name) + run_idx
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(
        lambda trial: objective(trial, X_opt, y_opt, batch_size),
        n_trials=OPT_TRIALS,
        timeout=OPT_TIMEOUT,
    )

    try:
        best_params = study.best_params
    except ValueError:
        nan_arr = np.full(n_eval_batches, np.nan)
        return nan_arr, nan_arr.copy()

    model = InstrumentedMicroTEDAClus(**best_params)
    df = evaluate_instrumented(model, X_eval, y_eval, batch_size)

    total_time_ms = df[TIMER_KEYS].sum(axis=1).to_numpy()
    return df["microcluster_count"].to_numpy(), total_time_ms


def run_stage(stage_name, X, y):
    n_samples = len(X)
    split_idx = int(SPLIT_FRAC * n_samples)
    X_opt, y_opt = X[:split_idx], y[:split_idx]
    X_eval, y_eval = X[split_idx:], y[split_idx:]
    n_eval_batches = len(X_eval) // BATCH_SIZE

    mc_runs, tt_runs = [], []
    for run_idx in tqdm(range(N_RUNS), desc=f"{stage_name} runs", leave=False):
        mc, tt = tune_and_evaluate_once(
            stage_name, run_idx, X_opt, y_opt, X_eval, y_eval, BATCH_SIZE
        )
        mc_runs.append(mc)
        tt_runs.append(tt)

    if all(np.all(np.isnan(mc)) for mc in mc_runs):
        print(f"[{stage_name}] All trials pruned in every run — skipping.")
        return None

    x_vals = (np.arange(1, n_eval_batches + 1)) * BATCH_SIZE

    return {
        "x": x_vals,
        "microcluster_count": aggregate_curves(mc_runs),
        "total_time": aggregate_curves(tt_runs),
    }


def main():
    print("Loading chameleon ds7 (cluto-t7-10k)...")
    X0, y0 = get_chameleon(CHAMELEON_DS7_URL)

    X_nr, y_nr = op_noise_removed(X0, y0)
    X_ds, y_ds = op_downsampled(X_nr, y_nr)

    stage_datasets = {
        "noise_removed": (X_nr, y_nr),
        "downsampled": (X_ds, y_ds),
    }

    results = {}
    for stage_name in tqdm(STAGES, desc="Stages"):
        X, y = stage_datasets[stage_name]
        res = run_stage(stage_name, X, y)
        if res is not None:
            results[stage_name] = res

    if len(results) == len(STAGES):
        plot_stage_comparisons(results, OUTPUT_DIR)
    else:
        print("Skipping comparison plots — not all stages produced results.")

    print(f"Done. Plots written to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
