import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import optuna
from sklearn.metrics import adjusted_rand_score

from EvolvingMicroTEDA import EvolvingMicroTEDA

DATASETS_PARAMS = {
    "chameleon-ds7-clean": {"variance_limit": 0.0007, "decay": np.nan},
    "chameleon-ds8-clean": {"variance_limit": 0.0003, "decay": np.nan},
}


def tune_hyperparameters(X, y_true, first_guess_variance_limit):
    def objective(trial):
        variance_limit = trial.suggest_float("variance_limit", 0.0001, 0.001)

        model = EvolvingMicroTEDA(variance_limit=variance_limit, decay=len(y_true)) # see (1)
        model.fit(X)
        y_pred = model.predict(X)
        score = adjusted_rand_score(y_true, y_pred)
        return score

    study = optuna.create_study(direction="maximize")
    study.enqueue_trial(
        {"variance_limit": first_guess_variance_limit}
    )
    study.optimize(objective, n_trials=64, timeout=300)
    return study.best_params


def main():

    results_global = []

    for dataset, first_guess_params in DATASETS_PARAMS.items():
        print(f"Processing dataset: {dataset}")
        data = pd.read_csv(f"{dataset}.csv")
        X = data.drop(columns=["class"]).to_numpy()
        y_true = data["class"].to_numpy()

        best_params = tune_hyperparameters(
            X, y_true, first_guess_params["variance_limit"]
        )

        print(f"Best parameters for {dataset}: {best_params}")
        model = EvolvingMicroTEDA(
            variance_limit=best_params["variance_limit"], decay=len(y_true)) # 1: Juan artificially cleared
                                                                                    # the datasets to remove noise
                                                                                    # which is totally dishonest
                                                                                    # but will be repeated here
                                                                                    # to have a baseline
                                                                                    # Because of that and the fact
                                                                                    # that those are static datasets, 
                                                                                    # the best decay is the one that
                                                                                    # keeps all the data
                                                                                    # Unfortunately, the implementation
                                                                                    # is also very slow,
                                                                                    # so each trial takes a long time

        fit_start = time.perf_counter()
        model.fit(X)
        fit_time = time.perf_counter() - fit_start
        print(f"Fit time for {dataset}: {fit_time:.4f} seconds")

        predict_start = time.perf_counter()
        y_pred = model.predict(X)
        predict_time = time.perf_counter() - predict_start
        print(f"Predict time for {dataset}: {predict_time:.4f} seconds")

        score = adjusted_rand_score(y_true, y_pred)
        print(f"Adjusted Rand Score for {dataset}: {score}")

        plt.figure()
        plt.scatter(X[:, 0], X[:, 1], c=y_pred)
        plt.title(f"{dataset} - Adjusted Rand Score: {score:.4f}")
        plt.savefig(f"{dataset}_predictions.png")
        plt.close()

        results_global.append({
            "dataset": dataset,
            "ari_score": score,
            "variance_limit": best_params["variance_limit"],
            "fit_time": fit_time,
            "predict_time": predict_time
        })

    # Convert results to DataFrame and save to CSV file
    final_results = pd.DataFrame(results_global)
    final_results.to_csv("final_results.csv", index=False, mode="a")

if __name__ == "__main__":
    main()
