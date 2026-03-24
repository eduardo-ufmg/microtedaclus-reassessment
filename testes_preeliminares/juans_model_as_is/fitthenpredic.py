import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import optuna
from sklearn.metrics import adjusted_rand_score

from EvolvingMicroTEDA import EvolvingMicroTEDA

DATASETS_PARAMS = {
    "chamaleon-ds7-clean": {"variance_limit": 0.0007, "decay": 2000},
    "chamaleon-ds8-clean": {"variance_limit": 0.0003, "decay": 2000},
}


def tune_hyperparameters(X, y_true, first_guess_variance_limit, first_guess_decay):
    def objective(trial):
        variance_limit = trial.suggest_float("variance_limit", 0.0001, 0.001)
        decay = trial.suggest_int("decay", 512, 4096, log=True)

        model = EvolvingMicroTEDA(variance_limit=variance_limit, decay=decay)
        model.fit(X)
        y_pred = model.predict(X)
        score = adjusted_rand_score(y_true, y_pred)
        return score

    study = optuna.create_study(direction="maximize")
    study.enqueue_trial(
        {"variance_limit": first_guess_variance_limit, "decay": first_guess_decay}
    )
    study.optimize(objective, n_trials=64, timeout=300)
    return study.best_params


def main():
    for dataset, first_guess_params in DATASETS_PARAMS.items():
        print(f"Processing dataset: {dataset}")
        data = pd.read_csv(f"{dataset}.csv")
        X = data.drop(columns=["class"]).to_numpy()
        y_true = data["class"].to_numpy()

        best_params = tune_hyperparameters(
            X, y_true, first_guess_params["variance_limit"], first_guess_params["decay"]
        )

        print(f"Best parameters for {dataset}: {best_params}")
        model = EvolvingMicroTEDA(
            variance_limit=best_params["variance_limit"], decay=best_params["decay"]
        )
        model.fit(X)
        y_pred = model.predict(X)
        score = adjusted_rand_score(y_true, y_pred)
        print(f"Adjusted Rand Score for {dataset}: {score}")

        plt.figure()
        plt.scatter(X[:, 0], X[:, 1], c=y_pred)
        plt.title(f"{dataset} - Adjusted Rand Score: {score:.4f}")
        plt.savefig(f"{dataset}_predictions.png")
        plt.close()


if __name__ == "__main__":
    main()
