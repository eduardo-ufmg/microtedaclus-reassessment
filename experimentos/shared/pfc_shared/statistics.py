"""Ingredients of the statistical analysis: discretization per metric, paired test and effect size."""

import warnings

import numpy as np
from scipy.stats import wilcoxon

ALPHA = 0.05
CLIFFS_DELTA_THRESHOLD = 0.147

METRICS = {
    "ari_overall": {
        "dir": "max",
        "round": lambda x: np.round(x, 3),
        "fmt": "{:.3f}",
    },
    "ari_steadystate": {
        "dir": "max",
        "round": lambda x: np.round(x, 3),
        "fmt": "{:.3f}",
    },
    "ari_fifth": {"dir": "max", "round": lambda x: np.round(x, 3), "fmt": "{:.3f}"},
    "ari_diff": {"dir": "min", "round": lambda x: np.round(x, 3), "fmt": "{:.3f}"},
    "time_predict_cum_ms": {
        "dir": "min",
        "round": lambda x: np.round(x, -1),
        "fmt": "{:.1f}",
    },
    "time_fit_cum_ms": {
        "dir": "min",
        "round": lambda x: np.round(x, -1),
        "fmt": "{:.1f}",
    },
    "memory_max_bytes": {
        "dir": "min",
        "round": lambda x: np.round(x / 1024) * 1024,
        "fmt": "{:.0f}",
    },
    "memory_steadystate_bytes": {
        "dir": "min",
        "round": lambda x: np.round(x / 1024) * 1024,
        "fmt": "{:.0f}",
    },
    "mc_max": {"dir": "min", "round": lambda x: np.round(x), "fmt": "{:.0f}"},
    "mc_steadystate": {"dir": "min", "round": lambda x: np.round(x), "fmt": "{:.0f}"},
}


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    mat = np.sign(x[:, None] - y[None, :])
    return float(np.mean(mat))


def safe_wilcoxon(x: np.ndarray, y: np.ndarray) -> float:
    diff = x - y
    if np.all(diff == 0):
        return 1.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return float(wilcoxon(x, y).pvalue)
        except ValueError:
            return 1.0
