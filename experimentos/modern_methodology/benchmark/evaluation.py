"""Prequential evaluation: every batch is predicted before it is learned."""

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

BATCH_SIZE = 100


def prequential(model, X, y, batch_size=BATCH_SIZE) -> pd.DataFrame:
    """ARI and summary size of the model at every batch of the stream."""
    rows = []
    for batch in range(len(X) // batch_size):
        rows_of_batch = slice(batch * batch_size, (batch + 1) * batch_size)
        predictions = model.predict(X[rows_of_batch])
        model.learn(X[rows_of_batch])
        rows.append(
            {
                "batch": batch,
                "ari": adjusted_rand_score(y[rows_of_batch], predictions),
                "summary_size": model.summary_size(),
            }
        )
    return pd.DataFrame(rows)


def ari_summary(results: pd.DataFrame) -> dict:
    """The two decision metrics of the thesis: Overall and Steady-State ARI."""
    ari = results["ari"].to_numpy()
    return {"ari_overall": float(np.median(ari)), "ari_steadystate": float(ari[-2])}
