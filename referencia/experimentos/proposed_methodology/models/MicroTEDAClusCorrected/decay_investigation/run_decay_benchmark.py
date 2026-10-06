"""Diagnostic Benchmarking Harness for Stage 4 Decay Logic Candidates.

Compares:
1. MicroTEDAClusCorrectedTypi (Stage 3 Control)
2. MicroTEDAClusCorrectedDecay (Stage 4 Reference - Inertia Decay v = v - rho/S)
3. MicroTEDAClusCorrectedDecayLinear (Candidate 1 - Constant Penalty v = v - rho)
4. MicroTEDAClusCorrectedDecayBounded (Candidate 2 - Saturated Inertia v = v - rho/min(S, 10))
5. MicroTEDAClusCorrectedDecayRecency (Candidate 3 - Timestamp Inactivity Window)

Evaluates on:
- chameleon_ds3 (10,000 samples - computational bottleneck)
- cassini (4,000 samples - quality sensitivity)
Across seeds: [0, 1, 2]
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from scipy.io import arff
from sklearn.metrics import adjusted_rand_score

# Set MPLCONFIGDIR to avoid warnings
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-decay-bench")

# Ensure repository root is on path
CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from main_experiment import get_dataset
from models.MicroTEDAClusCorrected.MicroTEDAClusCorrected_decay import (
    MicroTEDAClusCorrectedDecay,
    MicroTEDAClusCorrectedDecayParameterSearchSpace,
)
from models.MicroTEDAClusCorrected.MicroTEDAClusCorrected_typi import (
    MicroTEDAClusCorrectedTypi,
    MicroTEDAClusCorrectedTypiParameterSearchSpace,
)
from models.MicroTEDAClusCorrected.decay_investigation.MicroTEDAClusCorrected_decay_bounded import (
    MicroTEDAClusCorrectedDecayBounded,
    MicroTEDAClusCorrectedDecayBoundedParameterSearchSpace,
)
from models.MicroTEDAClusCorrected.decay_investigation.MicroTEDAClusCorrected_decay_linear import (
    MicroTEDAClusCorrectedDecayLinear,
    MicroTEDAClusCorrectedDecayLinearParameterSearchSpace,
)
from models.MicroTEDAClusCorrected.decay_investigation.MicroTEDAClusCorrected_decay_recency import (
    MicroTEDAClusCorrectedDecayRecency,
    MicroTEDAClusCorrectedDecayRecencyParameterSearchSpace,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)

BENCHMARK_MODELS = {
    "typi (control)": (
        MicroTEDAClusCorrectedTypi,
        MicroTEDAClusCorrectedTypiParameterSearchSpace,
        "MicroTEDAClusCorrected_typi",
    ),
    "decay (reference)": (
        MicroTEDAClusCorrectedDecay,
        MicroTEDAClusCorrectedDecayParameterSearchSpace,
        "MicroTEDAClusCorrected_decay",
    ),
    "decay_linear (cand 1)": (
        MicroTEDAClusCorrectedDecayLinear,
        MicroTEDAClusCorrectedDecayLinearParameterSearchSpace,
        "MicroTEDAClusCorrected_decay_linear",
    ),
    "decay_bounded (cand 2)": (
        MicroTEDAClusCorrectedDecayBounded,
        MicroTEDAClusCorrectedDecayBoundedParameterSearchSpace,
        "MicroTEDAClusCorrected_decay_bounded",
    ),
    "decay_recency (cand 3)": (
        MicroTEDAClusCorrectedDecayRecency,
        MicroTEDAClusCorrectedDecayRecencyParameterSearchSpace,
        "MicroTEDAClusCorrected_decay_recency",
    ),
}

BENCHMARK_DATASETS = ["chameleon_ds3", "cassini"]
SEEDS = [0, 1, 2]
BATCH_SIZE = 100
SPLIT_FRACTION = 0.2
BURN_IN_FRACTION = 0.05
OPT_TRIALS = 25
OPT_TIMEOUT = 60

DIR_PARAMS_EXP = REPO_ROOT / "results_params_experiments"
DIR_PARAMS_CACHE = CURRENT_DIR / "params_cache"
DIR_PARAMS_CACHE.mkdir(exist_ok=True)
DIR_BENCH_RESULTS = CURRENT_DIR / "results"
DIR_BENCH_RESULTS.mkdir(exist_ok=True)


def load_dataset(dataset_name: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Load dataset, using local cached arff files when available."""
    if dataset_name == "chameleon_ds3":
        local_path = REPO_ROOT / "data_samples" / "cluto-t7-10k.arff"
        if local_path.exists():
            data, _ = arff.loadarff(str(local_path))
            df = pd.DataFrame(data)
            target_col = df.columns[-1]
            y_raw = df[target_col].str.decode("utf-8").replace("noise", "-1")
            X = df.drop(columns=[target_col]).to_numpy()
            y = y_raw.astype(int).to_numpy()
            return X, y
    return get_dataset(dataset_name, seed)


def get_or_tune_params(
    dataset: str,
    model_tag: str,
    model_class: type,
    search_space_class: type,
    run: int,
    X_opt: np.ndarray,
    y_opt: np.ndarray,
) -> dict:
    """Retrieve existing tuned parameters or run Optuna search."""
    # 1. Check official experiments params directory
    official_param_file = DIR_PARAMS_EXP / f"{dataset}_{model_tag}_{run}.csv"
    if official_param_file.exists():
        df_p = pd.read_csv(official_param_file)
        params = dict(zip(df_p["param_name"], df_p["param_value"]))
        if "decay" in params:
            params["decay"] = int(params["decay"])
        return params

    # 2. Check local benchmark cache
    cache_param_file = DIR_PARAMS_CACHE / f"{dataset}_{model_tag}_{run}.csv"
    if cache_param_file.exists():
        df_p = pd.read_csv(cache_param_file)
        params = dict(zip(df_p["param_name"], df_p["param_value"]))
        if "decay" in params:
            params["decay"] = int(params["decay"])
        return params

    # 3. Optimize via Optuna
    def objective(trial):
        p = search_space_class().sample_parameters(trial)
        try:
            m = model_class(**p)
        except Exception:
            raise optuna.exceptions.TrialPruned()

        burn_in_idx = int(BURN_IN_FRACTION * len(X_opt))
        y_pred_all = []
        num_batches = len(X_opt) // BATCH_SIZE
        t0 = time.process_time_ns()

        try:
            for b in range(num_batches):
                s_idx, e_idx = b * BATCH_SIZE, (b + 1) * BATCH_SIZE
                xb = X_opt[s_idx:e_idx]
                preds = m.predict(xb)
                y_pred_all.extend(preds)
                m.fit(xb)
        except Exception:
            raise optuna.exceptions.TrialPruned()

        t_total_sec = (time.process_time_ns() - t0) / 1e9
        y_pred_eval = np.array(y_pred_all)[burn_in_idx:]
        y_opt_eval = y_opt[: len(y_pred_all)][burn_in_idx:]
        ari = adjusted_rand_score(y_opt_eval, y_pred_eval)
        return ari - (1e-3 * t_total_sec)

    sampler = optuna.samplers.TPESampler(seed=run)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=OPT_TRIALS, timeout=OPT_TIMEOUT)

    best_p = study.best_params
    if "decay" in best_p:
        best_p["decay"] = int(best_p["decay"])

    # Save to local cache
    rec = [
        {
            "dataset": dataset,
            "model": model_tag,
            "run": run,
            "param_name": k,
            "param_value": v,
        }
        for k, v in best_p.items()
    ]
    pd.DataFrame(rec).to_csv(cache_param_file, index=False)
    return best_p


def evaluate_model(
    model_class: type,
    params: dict,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
) -> dict:
    """Evaluate model on prequential stream batches and record diagnostics."""
    model = model_class(**params)
    num_batches = len(X_eval) // BATCH_SIZE
    aris = []
    mc_counts = []
    total_fit_time = 0.0

    for b in range(num_batches):
        s_idx, e_idx = b * BATCH_SIZE, (b + 1) * BATCH_SIZE
        xb, yb = X_eval[s_idx:e_idx], y_eval[s_idx:e_idx]

        preds = model.predict(xb)
        t0 = time.perf_counter()
        model.fit(xb)
        total_fit_time += time.perf_counter() - t0

        ari = adjusted_rand_score(yb, preds)
        aris.append(ari)
        mc_counts.append(model.microcluster_count())

    return {
        "fit_time_sec": total_fit_time,
        "mc_max": max(mc_counts) if mc_counts else 0,
        "mc_ss": mc_counts[-1] if mc_counts else 0,
        "mean_ari": float(np.mean(aris)) if aris else 0.0,
    }


def main():
    print("=" * 80)
    print("STAGE 4 DECAY LOGIC INVESTIGATION & DIAGNOSTIC BENCHMARK")
    print("=" * 80)

    records = []

    for dataset in BENCHMARK_DATASETS:
        print(f"\nEvaluating Dataset: {dataset.upper()}")
        print("-" * 80)

        for model_label, (model_cls, space_cls, model_tag) in BENCHMARK_MODELS.items():
            run_metrics = []

            for seed in SEEDS:
                X, y = load_dataset(dataset, seed)
                split_idx = int(SPLIT_FRACTION * len(X))
                X_opt, y_opt = X[:split_idx], y[:split_idx]
                X_eval, y_eval = X[split_idx:], y[split_idx:]

                t_start = time.time()
                params = get_or_tune_params(
                    dataset, model_tag, model_cls, space_cls, seed, X_opt, y_opt
                )
                tune_elapsed = time.time() - t_start

                m_res = evaluate_model(model_cls, params, X_eval, y_eval)
                m_res["seed"] = seed
                m_res["tune_time"] = tune_elapsed
                m_res["params"] = str(params)
                run_metrics.append(m_res)

                print(
                    f"  [{model_label:22s}] Seed {seed}: "
                    f"Fit={m_res['fit_time_sec']:6.2f}s | "
                    f"McMax={m_res['mc_max']:4d} | "
                    f"McSS={m_res['mc_ss']:4d} | "
                    f"ARI={m_res['mean_ari']:6.3f} | "
                    f"decay={params.get('decay', 'N/A')}"
                )

            # Compute aggregate summary across seeds
            fit_times = [r["fit_time_sec"] for r in run_metrics]
            mc_maxs = [r["mc_max"] for r in run_metrics]
            mc_sss = [r["mc_ss"] for r in run_metrics]
            aris = [r["mean_ari"] for r in run_metrics]

            records.append(
                {
                    "dataset": dataset,
                    "model": model_label,
                    "fit_time_mean": np.mean(fit_times),
                    "fit_time_std": np.std(fit_times),
                    "mc_max_mean": np.mean(mc_maxs),
                    "mc_max_std": np.std(mc_maxs),
                    "mc_ss_mean": np.mean(mc_sss),
                    "mc_ss_std": np.std(mc_sss),
                    "ari_mean": np.mean(aris),
                    "ari_std": np.std(aris),
                }
            )

    # Save summary dataframe
    df_summary = pd.DataFrame(records)
    summary_path = DIR_BENCH_RESULTS / "decay_benchmark_summary.csv"
    df_summary.to_csv(summary_path, index=False)

    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY RESULTS TABLE")
    print("=" * 80)

    for ds in BENCHMARK_DATASETS:
        df_ds = df_summary[df_summary["dataset"] == ds]
        print(f"\n### Dataset: {ds}")
        print("| Model | Fit Time (s) | Mc Max | Mc Steady-State | Prequential ARI |")
        print("| :--- | :---: | :---: | :---: | :---: |")
        for _, row in df_ds.iterrows():
            print(
                f"| **{row['model']}** | "
                f"{row['fit_time_mean']:.2f} ± {row['fit_time_std']:.2f} | "
                f"{row['mc_max_mean']:.0f} ± {row['mc_max_std']:.0f} | "
                f"{row['mc_ss_mean']:.0f} ± {row['mc_ss_std']:.0f} | "
                f"**{row['ari_mean']:.3f} ± {row['ari_std']:.3f}** |"
            )

    print(f"\nFull summary CSV saved to: {summary_path}")


if __name__ == "__main__":
    main()
