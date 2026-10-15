"""Clustering models behind one interface.

A model learns a batch, predicts the labels of a batch, with -1 for noise and
for any sample predicted before the model has clusters, and reports the size of
its summary of the stream, such as its number of microclusters. It declares the
category of every hyperparameter it exposes, and whether it takes a seed.
"""

from benchmark.models.bico import Bico
from benchmark.models.microtedaclus_plus import MicroTEDAClusPlus
from benchmark.models.river import DBSTREAM, DenStream
from benchmark.models.scikit import Birch, MiniBatchKMeans

MODELS = {
    "MicroTEDAClusPlus": MicroTEDAClusPlus,
    "DenStream": DenStream,
    "DBSTREAM": DBSTREAM,
    "BIRCH": Birch,
    "MiniBatchKMeans": MiniBatchKMeans,
    "BICO": Bico,
}


def build(model_class, hyperparameters: dict, seed: int):
    """A model with the given hyperparameters; the seed goes to the models that take one."""
    if getattr(model_class, "SEEDED", False):
        return model_class(**hyperparameters, seed=seed)
    return model_class(**hyperparameters)
