import time

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from EvolvingMicroTEDA import EvolvingMicroTEDA
from sklearn.metrics import adjusted_rand_score

DATASETS = {
    "dynamic_rotating_blobs": (0.0001, 5.0),
}

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


def tune_hyperparameters(
    X: np.ndarray, y_true: np.ndarray, tune_range: tuple[float, float]
):
    n_samples = len(X)

    def objective(trial):
        variance_limit = trial.suggest_float(
            "variance_limit", tune_range[0], tune_range[1], log=True
        )
        decay = trial.suggest_float("decay", float(n_samples) / 20, float(n_samples))

        model = EvolvingMicroTEDA(variance_limit=variance_limit, decay=decay)
        model.fit(X)
        y_pred = model.predict(X)
        score = adjusted_rand_score(y_true, y_pred)
        return score

    study = optuna.create_study(direction="maximize")

    study.optimize(objective, n_trials=64, timeout=len(y_true) * 100 / 1000)
    return study.best_params


def run_offline(X: np.ndarray, y: np.ndarray, variance_limit: float, decay: float):
    model = EvolvingMicroTEDA(variance_limit=variance_limit, decay=decay)
    n_samples = len(X)

    fit_start = time.perf_counter()
    model.fit(X)
    total_fit_time = time.perf_counter() - fit_start

    predict_start = time.perf_counter()
    y_pred = model.predict(X)
    total_pred_time = time.perf_counter() - predict_start

    score = adjusted_rand_score(y, y_pred)

    return score, y_pred, total_fit_time / n_samples, total_pred_time / n_samples


def main():
    headers = [
        "dataset",
        "best_variance_limit",
        "best_decay",
        "tune_ari_global",
        "test_ari_global",
        "test_fit_lat_sec",
        "test_pred_lat_sec",
    ]
    pd.DataFrame(columns=headers).to_csv(
        "fitthenpredict_results.csv", index=False, mode="a"
    )

    for dataset, tune_range in DATASETS.items():
        print(f"Processing dataset: {dataset} (Offline Baseline)")
        data = pd.read_csv(f"{dataset}.csv")
        X = data.drop(columns=["y"]).to_numpy()
        y_true = data["y"].to_numpy()

        (X_tune, y_tune), (X_eval, y_eval) = get_subsets(
            X, y_true, TUNE_SUBSET_SIZE_FACTOR
        )

        best_params = tune_hyperparameters(X_tune, y_tune, tune_range)

        tune_score, _, _, _ = run_offline(
            X_tune, y_tune, best_params["variance_limit"], best_params["decay"]
        )

        print(f"Best parameters for {dataset}: {best_params}")

        test_score, y_pred_test, fit_lat, pred_lat = run_offline(
            X_eval, y_eval, best_params["variance_limit"], best_params["decay"]
        )

        print(f"Test Fit Latency: {fit_lat:.6f} sec/sample")
        print(f"Test Predict Latency: {pred_lat:.6f} sec/sample")
        print(f"Test Adjusted Rand Score: {test_score:.4f}")

        plt.figure()
        plt.scatter(X_eval[:, 0], X_eval[:, 1], c=y_pred_test)
        plt.title(f"{dataset} (Eval) - ARI: {test_score:.4f}")
        plt.savefig(f"{dataset}_eval_predictions.png")
        plt.close()

        results_df = pd.DataFrame(
            [
                {
                    "dataset": dataset,
                    "best_variance_limit": best_params["variance_limit"],
                    "best_decay": best_params["decay"],
                    "tune_ari_global": tune_score,
                    "test_ari_global": test_score,
                    "test_fit_lat_sec": fit_lat,
                    "test_pred_lat_sec": pred_lat,
                }
            ]
        )

        results_df.to_csv(
            "fitthenpredict_results.csv", index=False, header=False, mode="a"
        )


if __name__ == "__main__":
    main()
