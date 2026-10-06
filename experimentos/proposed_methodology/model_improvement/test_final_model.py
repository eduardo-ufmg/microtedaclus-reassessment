"""The standalone final model must behave as the engine configuration it was distilled from.

Run with `python -m pytest model_improvement/test_final_model.py` or
`python -m model_improvement.test_final_model` (from experimentos/proposed_methodology).
"""

import numpy as np

from model_improvement.experiments import FINAL
from model_improvement.variants import resolve
from models.MicroTEDAClusPlus.MicroTEDAClusPlus import MicroTEDAClusPlus
from streams import generate_scl

CASES = [
    (2, dict(variance_limit=0.02, decay=60, noise_distance=4.0)),
    (2, dict(variance_limit=0.002, decay=800, noise_distance=7.5)),
    (8, dict(variance_limit=0.05, decay=200, noise_distance=5.0)),
    (32, dict(variance_limit=0.1, decay=1000, noise_distance=6.0)),
]


def test_matches_engine_configuration():
    engine_class, _ = resolve(FINAL)
    for dim, params in CASES:
        X, _ = generate_scl(dim, seed=dim)
        engine = engine_class(
            variance_limit=params["variance_limit"],
            decay=params["decay"],
            noise_m=params["noise_distance"],
        )
        model = MicroTEDAClusPlus(**params)
        for start in range(0, 2000, 100):
            batch = X[start : start + 100]
            assert np.array_equal(model.predict(batch), engine.predict(batch))
            model.fit(batch)
            engine.fit(batch)
            assert model.microcluster_count() == engine.microcluster_count()


if __name__ == "__main__":
    test_matches_engine_configuration()
    print("the standalone model matches the engine configuration")
