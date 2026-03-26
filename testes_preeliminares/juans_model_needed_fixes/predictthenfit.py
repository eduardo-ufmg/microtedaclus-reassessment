import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import optuna
import os
from sklearn.metrics import adjusted_rand_score

from EvolvingMicroTEDA import EvolvingMicroTEDA

DATASETS = {"chameleon-ds7-clean", "chameleon-ds8-clean"}

BATCH_SIZES = [32, 64, 128, 256, 512, 1024]

TUNE_SUBSET_SIZE_FACTOR = 0.5


def get_subsets(
    X: np.ndarray, y: np.ndarray, factor: float, random_state: int = 0
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """
    Select two random subsets of the data. Sample order is preserved.
    """
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
    """
    Do the following:
    1. Instantiate the model with the given parameters.
    2. Split the data into batches of the given size, except for the last batch which may be smaller.
    3. Fit the model on the first batch.
    4. For each subsequent batch, predict the labels, compute the ARI score, and then fit the model on the batch.

    Parameters:
    - X: The input data.
    - y: The true labels.
    - batch_size: The size of each batch.
    - variance_limit: The variance limit for the model.
    - decay: The decay factor for the model.

    Returns:
    - ari_wm: Weighted mean ARI score across all batches. w_b = n_b / n_total.
    - fit_t_wm: Weighted mean fit time across all batches. w_b = n_b / n_total.
    - pred_t_wm: Weighted mean prediction time across all batches. w_b = n_b / n_total.
    """

    model = EvolvingMicroTEDA(variance_limit=variance_limit, decay=decay)

    n_samples = X.shape[0]
    n_batches = (n_samples + batch_size - 1) // batch_size

    X_fb, y_fb = X[:batch_size], y[:batch_size]

    model.fit(X_fb)

    X, y = X[batch_size:], y[batch_size:]

    ari_scores = []
    fit_times = []
    pred_times = []

    for i in range(n_batches - 1):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_samples - batch_size)

        X_batch, y_batch = X[start_idx:end_idx], y[start_idx:end_idx]

        start_time = time.time()
        y_pred = model.predict(X_batch)
        pred_time = time.time() - start_time

        ari_score = adjusted_rand_score(y_batch, y_pred)

        start_time = time.time()
        model.fit(X_batch)
        fit_time = time.time() - start_time

        ari_scores.append(ari_score)
        fit_times.append(fit_time)
        pred_times.append(pred_time)

    w = [
        len(X_batch) / n_samples
        for X_batch in [X[start_idx:end_idx] for i in range(n_batches - 1)]
    ]

    ari_wm = np.sum(np.array(ari_scores) * np.array(w))
    fit_t_wm = np.sum(np.array(fit_times) * np.array(w))
    pred_t_wm = np.sum(np.array(pred_times) * np.array(w))

    return ari_wm, fit_t_wm, pred_t_wm


def tune_hyperparameters(
    X: np.ndarray, y: np.ndarray, batch_size: int
) -> tuple[float, float, float, float]:
    """
    Tune the variance_limit and decay hyperparameters using Optuna.

    Parameters:
    - X: The input data.
    - y: The true labels.
    - batch_size: The size of each batch.

    Returns:
    - best_variance_limit: The best variance limit found.
    - tune_ari_wm: The best weighted mean ARI score achieved with the best hyperparameters.
    - tune_fit_t_wm: The weighted mean fit time achieved with the best hyperparameters.
    - tune_pred_t_wm: The weighted mean prediction time achieved with the best hyperparameters.
    """

    def objective(trial):
        variance_limit = trial.suggest_float("variance_limit", 0.0001, 0.1)

        ari_wm, fit_t_wm, pred_t_wm = run(
            X, y, batch_size, variance_limit, decay=batch_size
        )

        return ari_wm

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=64, timeout=600)

    best_variance_limit = study.best_params["variance_limit"]

    tune_ari_wm, tune_fit_t_wm, tune_pred_t_wm = run(
        X, y, batch_size, best_variance_limit, decay=batch_size
    )

    return best_variance_limit, tune_ari_wm, tune_fit_t_wm, tune_pred_t_wm


def process_batch(
    dataset_name: str, batch_size: int, result_file: str = "predictthenfit_results.csv"
) -> None:
    """
    Process the given dataset with the specified batch size, tuning hyperparameters and evaluating the model.

    Parameters:
    - dataset_name: The name of the dataset to process.
    - batch_size: The size of each batch.

    Returns:
    - None
    """

    print(f"Processing dataset: {dataset_name}, batch size: {batch_size}")

    df = pd.read_csv(f"{dataset_name}.csv")
    X = df.drop(columns=["y"]).to_numpy()
    y = df["y"].to_numpy()

    (X_tune, y_tune), (X_eval, y_eval) = get_subsets(X, y, TUNE_SUBSET_SIZE_FACTOR)

    best_variance_limit, tune_ari_wm, tune_fit_t_wm, tune_pred_t_wm = (
        tune_hyperparameters(X_tune, y_tune, batch_size)
    )

    test_ari_wm, test_fit_t_wm, test_pred_t_wm = run(
        X_eval, y_eval, batch_size, best_variance_limit, decay=batch_size
    )

    results_df = pd.DataFrame(
        {
            "dataset": [dataset_name],
            "batch_size": [batch_size],
            "best_variance_limit": [best_variance_limit],
            "fixed_decay": [batch_size],
            "tune_ari_wm": [tune_ari_wm],
            "tune_fit_t_wm": [tune_fit_t_wm],
            "tune_pred_t_wm": [tune_pred_t_wm],
            "test_ari_wm": [test_ari_wm],
            "test_fit_t_wm": [test_fit_t_wm],
            "test_pred_t_wm": [test_pred_t_wm],
        }
    )

    results_df.to_csv(result_file, mode="a", header=False, index=False)

    print(f"Finished processing dataset: {dataset_name}, batch size: {batch_size}")


def process_dataset(
    dataset_name: str, result_file: str = "predictthenfit_results.csv"
) -> None:
    """
    Process the given dataset for all batch sizes and aggregate the results.

    Parameters:
    - dataset_name: The name of the dataset to process.

    Returns:
    - None
    """

    for batch_size in BATCH_SIZES:
        process_batch(dataset_name, batch_size, result_file)


def main():

    header = [
        "dataset",
        "batch_size",
        "best_variance_limit",
        "best_decay",
        "tune_ari_wm",
        "tune_fit_t_wm",
        "tune_pred_t_wm",
        "test_ari_wm",
        "test_fit_t_wm",
        "test_pred_t_wm",
    ]

    pd.DataFrame(columns=header).to_csv(
        "predictthenfit_results.csv", index=False, mode="a"
    )

    for dataset_name in DATASETS:
        process_dataset(dataset_name, "predictthenfit_results.csv")


if __name__ == "__main__":
    main()
