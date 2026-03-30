import time
import optuna
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from sklearn.metrics import adjusted_rand_score

from EvolvingMicroTEDAClean import EvolvingMicroTEDAClean

DATASETS = [
    "static_moons",
    "dynamic_nonoverlap_movingblobs",
    "dynamic_rotating_blobs",
    "dynamic_crossing_movingblobs",
    "dynamic_birthdeath",
]
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


def infer_bounds(X: np.ndarray) -> tuple[tuple[float, float], tuple[int, int]]:
    n_samples = len(X)
    idx = np.random.choice(n_samples, min(500, n_samples), replace=False)
    X_sample = X[idx]

    distances = pdist(X_sample, metric="euclidean")
    distances = distances[distances > 1e-9]

    if len(distances) == 0:
        res_bounds = (1e-6, 1.0)
    else:
        res_bounds = (float(np.min(distances)), float(np.max(distances)))

    mem_bounds = (max(n_samples // 20, 10), n_samples)
    return res_bounds, mem_bounds


def run(
    X: np.ndarray, y: np.ndarray, batch_size: int, resolution: float, memory: int
) -> tuple[float, float, float]:
    model = EvolvingMicroTEDAClean(resolution=resolution, memory=memory)

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
    X: np.ndarray, y: np.ndarray, batch_size: int
) -> tuple[float, int, float, float, float]:
    res_bounds, mem_bounds = infer_bounds(X)

    def objective(trial):
        resolution = trial.suggest_float(
            "resolution", res_bounds[0], res_bounds[1], log=True
        )
        memory = trial.suggest_int("memory", mem_bounds[0], mem_bounds[1])

        ari, _, _ = run(X, y, batch_size, resolution, memory)
        return ari

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=64, timeout=600)

    best_resolution = study.best_params["resolution"]
    best_memory = study.best_params["memory"]

    tune_ari, tune_fit_lat, tune_pred_lat = run(
        X, y, batch_size, best_resolution, best_memory
    )

    return best_resolution, best_memory, tune_ari, tune_fit_lat, tune_pred_lat


def process_batch(
    dataset_name: str,
    batch_size: int,
    result_file: str = "predictthenfit_results.csv",
) -> None:
    print(f"Processing dataset: {dataset_name}, batch size: {batch_size} (Clean Model)")

    df = pd.read_csv(f"{dataset_name}.csv")
    X = df.drop(columns=["y"]).to_numpy()
    y = df["y"].to_numpy()

    (X_tune, y_tune), (X_eval, y_eval) = get_subsets(X, y, TUNE_SUBSET_SIZE_FACTOR)

    best_resolution, best_memory, tune_ari, tune_fit_lat, tune_pred_lat = (
        tune_hyperparameters(X_tune, y_tune, batch_size)
    )

    test_ari, test_fit_lat, test_pred_lat = run(
        X_eval, y_eval, batch_size, best_resolution, best_memory
    )

    results_df = pd.DataFrame(
        {
            "dataset": [dataset_name],
            "batch_size": [batch_size],
            "best_resolution": [best_resolution],
            "best_memory": [best_memory],
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
        "best_resolution",
        "best_memory",
        "tune_ari_chunk",
        "tune_fit_lat_sec",
        "tune_pred_lat_sec",
        "test_ari_chunk",
        "test_fit_lat_sec",
        "test_pred_lat_sec",
    ]

    pd.DataFrame(columns=header).to_csv(
        "predictthenfit_results.csv", index=False, mode="a"
    )

    for dataset_name in DATASETS:
        for batch_size in BATCH_SIZES:
            process_batch(dataset_name, batch_size)


if __name__ == "__main__":
    main()
