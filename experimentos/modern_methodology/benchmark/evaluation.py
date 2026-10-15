"""Prequential evaluation: every batch is predicted before it is learned."""

import time

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

BATCH_SIZE = 100
WARM_UP_BATCHES = 3  # batches that a model processes before its cost is measured


def batches(n_samples: int, batch_size=BATCH_SIZE):
    """Rows of the consecutive whole batches of a stream."""
    for batch in range(n_samples // batch_size):
        yield slice(batch * batch_size, (batch + 1) * batch_size)


def process(model, X, batch_size=BATCH_SIZE) -> None:
    """Predict and learn every batch of a stream, keeping nothing."""
    for rows_of_batch in batches(len(X), batch_size):
        model.predict(X[rows_of_batch])
        model.learn(X[rows_of_batch])


def warm_up(make_model, X, batches=WARM_UP_BATCHES, batch_size=BATCH_SIZE) -> None:
    """Let a new model process the first batches of a stream, or all of it, and discard it.

    Whatever the model loads or creates on its first use is then in place, so
    that a measurement that follows counts the model and not its first use.
    With `batches` None, the model processes the whole stream.
    """
    end = None if batches is None else batches * batch_size
    process(make_model(), X[:end], batch_size)


def prequential(model, X, y, batch_size=BATCH_SIZE, measure=False) -> pd.DataFrame:
    """ARI of the model at every batch of the stream.

    With `measure`, also the time of each step and the size of the summary
    after each batch, in the columns that the run-level metrics of the thesis
    read.
    """
    rows = []
    for batch, rows_of_batch in enumerate(batches(len(X), batch_size)):
        X_batch = X[rows_of_batch]

        wall, cpu = time.perf_counter_ns(), time.process_time_ns()
        predictions = model.predict(X_batch)
        predict_cpu, predict_wall = (
            time.process_time_ns() - cpu,
            time.perf_counter_ns() - wall,
        )

        wall, cpu = time.perf_counter_ns(), time.process_time_ns()
        model.learn(X_batch)
        learn_cpu, learn_wall = (
            time.process_time_ns() - cpu,
            time.perf_counter_ns() - wall,
        )

        row = {
            "batch": batch,
            "ari": adjusted_rand_score(y[rows_of_batch], predictions),
        }
        if measure:
            row |= {
                "time_predict_cpu_ns": predict_cpu,
                "time_fit_cpu_ns": learn_cpu,
                "time_predict_wall_ns": predict_wall,
                "time_fit_wall_ns": learn_wall,
                "microcluster_count": model.summary_size(),
            }
        rows.append(row)
    return pd.DataFrame(rows)


def ari_summary(results: pd.DataFrame) -> dict:
    """The two decision metrics of the thesis: Overall and Steady-State ARI."""
    ari = results["ari"].to_numpy()
    return {"ari_overall": float(np.median(ari)), "ari_steadystate": float(ari[-2])}
