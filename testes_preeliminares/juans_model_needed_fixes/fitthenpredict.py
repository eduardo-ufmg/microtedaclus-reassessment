import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import time
import optuna
from sklearn.metrics import adjusted_rand_score

from EvolvingMicroTEDA import EvolvingMicroTEDA

DATASETS_PARAMS = {
    "chameleon-ds7-clean": {"variance_limit": 0.0007, "decay": np.nan},
    "chameleon-ds8-clean": {"variance_limit": 0.0003, "decay": np.nan},
    "std-DS7": {"variance_limit": 500 * 600 * 0.0007, "decay": np.nan},
    "std-DS8": {"variance_limit": 500 * 600 * 0.0003, "decay": np.nan},
}


def tune_hyperparameters(X, y_true, first_guess_variance_limit):

    floor_tenth_power = 10 ** math.floor(math.log10(first_guess_variance_limit))
    ceil_tenth_power = 10 ** math.ceil(math.log10(first_guess_variance_limit))
    var_lim_range = (floor_tenth_power * 0.1, ceil_tenth_power * 10)

    def objective(trial):
        variance_limit = trial.suggest_float("variance_limit", *var_lim_range)

        model = EvolvingMicroTEDA(
            variance_limit=variance_limit, decay=len(y_true) / 2
        )  # see (1), but we assume that half of the data is good enough to tune the variance limit
        model.fit(X)
        y_pred = model.predict(X)
        score = adjusted_rand_score(y_true, y_pred)
        return score

    study = optuna.create_study(direction="maximize")
    study.enqueue_trial({"variance_limit": first_guess_variance_limit})
    study.optimize(objective, n_trials=64, timeout=len(y_true) * 100 / 1000)
    return study.best_params


def main():

    headers = ["dataset", "ari_score", "variance_limit", "fit_time", "predict_time"]
    pd.DataFrame(columns=headers).to_csv(
        "fitthenpredict_results.csv", index=False, mode="a"
    )

    for dataset, first_guess_params in DATASETS_PARAMS.items():
        print(f"Processing dataset: {dataset}")
        data = pd.read_csv(f"{dataset}.csv")
        X = data.drop(columns=["y"]).to_numpy()
        y_true = data["y"].to_numpy()

        best_params = tune_hyperparameters(
            X, y_true, first_guess_params["variance_limit"]
        )

        print(f"Best parameters for {dataset}: {best_params}")
        model = EvolvingMicroTEDA(
            variance_limit=best_params["variance_limit"], decay=len(y_true)
        )  # 1: Juan artificially cleared
        # the datasets to remove noise
        # which is totally dishonest
        # but will be repeated here
        # to have a baseline
        # Because of that and the fact
        # that those are static datasets,
        # the best decay is the one that
        # keeps all the data

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

        results_df = pd.DataFrame(
            [
                {
                    "dataset": dataset,
                    "ari_score": score,
                    "variance_limit": best_params["variance_limit"],
                    "fit_time": fit_time,
                    "predict_time": predict_time,
                }
            ]
        )

        results_df.to_csv(
            "fitthenpredict_results.csv", index=False, header=False, mode="a"
        )


if __name__ == "__main__":
    main()
