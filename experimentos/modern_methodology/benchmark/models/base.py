"""Interface of the clustering models of the experiment."""

from typing import Protocol

import numpy as np


class StreamClusterer(Protocol):
    def learn(self, X: np.ndarray) -> None:
        """Update the model with a batch of samples."""

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Labels of a batch of samples; -1 for noise."""

    def summary_size(self) -> int:
        """Number of summary structures the model keeps."""
