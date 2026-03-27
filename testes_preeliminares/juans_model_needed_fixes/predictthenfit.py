import time
import optuna
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from EvolvingMicroTEDA import EvolvingMicroTEDA

DATASETS = {
    "chameleon-ds7-clean": (0.0001, 0.01),
    "chameleon-ds8-clean": (0.0001, 0.01),
    "std-DS7": (10, 1000),
    "std-DS8": (10, 1000),
}

BATCH_SIZES = [32, 64, 128, 256, 512, 1024]
TUNE_SUBSET_SIZE_FACTOR = 0.5


def get_subsets(
    X: np.ndarray, y: np.ndarray, factor: float, random_state: int = 0
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    np.random.seed(random_state)
    n_samples = X.shape[0]
    subset_size = int(n_samples * factor)

    indices = np.random.choice(n_samples, size=subset_size, replace=False)

    X_subset = X[indices]
    y_subset = y[indices]

    X_remaining = np.delete(X, indices, axis=0)
    y_remaining = np.delete(y, indices, axis=0)

    return (X_subset, y_subset), (X_remaining, y_remaining)


def run(
    X: np.ndarray, y: np.ndarray, batch_size: int, variance_limit: float, decay: float
) -> tuple[float, float, float]:

    model = EvolvingMicroTEDA(variance_limit=variance_limit, decay=decay)

    n_samples = X.shape[0]
    if n_samples <= batch_size:
        return 0.0, 0.0, 0.0

    X_fb = X[:batch_size]
    model.fit(X_fb)

    X_eval, y_eval = X[batch_size:], y[batch_size:]
    n_eval_samples = len(X_eval)
    n_batches = int(np.ceil(n_eval_samples / batch_size))

    ari_scores = np.zeros(n_batches)
    batch_weights = np.zeros(n_batches)

    total_fit_time = 0.0
    total_pred_time = 0.0

    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_eval_samples)

        X_batch, y_batch = X_eval[start_idx:end_idx], y_eval[start_idx:end_idx]
        batch_len = len(X_batch)

        t0 = time.perf_counter()
        y_pred = model.predict(X_batch)
        total_pred_time += time.perf_counter() - t0

        t0 = time.perf_counter()
        model.fit(X_batch)
        total_fit_time += time.perf_counter() - t0

        ari_scores[i] = adjusted_rand_score(y_batch, y_pred)
        batch_weights[i] = batch_len

    ari_chunk_weighted = float(np.average(ari_scores, weights=batch_weights))
    fit_lat_avg = total_fit_time / n_eval_samples
    pred_lat_avg = total_pred_time / n_eval_samples

    return ari_chunk_weighted, fit_lat_avg, pred_lat_avg


def tune_hyperparameters(
    X: np.ndarray, y: np.ndarray, batch_size: int, tune_range: tuple[float, float]
) -> tuple[float, float, float, float, float]:

    n_samples = len(X)

    def objective(trial):
        variance_limit = trial.suggest_float(
            "variance_limit", tune_range[0], tune_range[1], log=True
        )
        decay = trial.suggest_float("decay", 1.0, float(n_samples))

        ari, _, _ = run(X, y, batch_size, variance_limit, decay)
        return ari

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=64, timeout=600)

    best_variance_limit = study.best_params["variance_limit"]
    best_decay = study.best_params["decay"]

    tune_ari, tune_fit_lat, tune_pred_lat = run(
        X, y, batch_size, best_variance_limit, best_decay
    )

    return best_variance_limit, best_decay, tune_ari, tune_fit_lat, tune_pred_lat


def process_batch(
    dataset_name: str,
    batch_size: int,
    tune_range: tuple[float, float],
    result_file: str = "predictthenfit_results.csv",
) -> None:
    print(
        f"Processing dataset: {dataset_name}, batch size: {batch_size} (Juan's Model)"
    )

    df = pd.read_csv(f"{dataset_name}.csv")
    X = df.drop(columns=["y"]).to_numpy()
    y = df["y"].to_numpy()

    (X_tune, y_tune), (X_eval, y_eval) = get_subsets(X, y, TUNE_SUBSET_SIZE_FACTOR)

    best_variance_limit, best_decay, tune_ari, tune_fit_lat, tune_pred_lat = (
        tune_hyperparameters(X_tune, y_tune, batch_size, tune_range)
    )

    test_ari, test_fit_lat, test_pred_lat = run(
        X_eval, y_eval, batch_size, best_variance_limit, best_decay
    )

    results_df = pd.DataFrame(
        {
            "dataset": [dataset_name],
            "batch_size": [batch_size],
            "best_variance_limit": [best_variance_limit],
            "best_decay": [best_decay],
            "tune_ari_chunk": [tune_ari],
            "tune_fit_lat_sec": [tune_fit_lat],
            "tune_pred_lat_sec": [tune_pred_lat],
            "test_ari_chunk": [test_ari],
            "test_fit_lat_sec": [test_fit_lat],
            "test_pred_lat_sec": [test_pred_lat],
        }
    )

    results_df.to_csv(result_file, mode="a", header=False, index=False)


def main():
    header = [
        "dataset",
        "batch_size",
        "best_variance_limit",
        "best_decay",
        "tune_ari_chunk",
        "tune_fit_lat_sec",
        "tune_pred_lat_sec",
        "test_ari_chunk",
        "test_fit_lat_sec",
        "test_pred_lat_sec",
    ]

    pd.DataFrame(columns=header).to_csv(
        "predictthenfit_results.csv", index=False, mode="w"
    )

    for dataset_name, tune_range in DATASETS.items():
        for batch_size in BATCH_SIZES:
            process_batch(dataset_name, batch_size, tune_range)


if __name__ == "__main__":
    main()
