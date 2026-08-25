#!/usr/bin/env python3
import os
import sys
import urllib.request
import logging
from io import StringIO
import traceback

# 1. Verify Core Python Dependencies
try:
    import numpy as np
    import pandas as pd
    import optuna
    from scipy.io import arff
    from sklearn.datasets import make_blobs
    from sklearn.metrics import adjusted_rand_score
    from pympler import asizeof
    import zlib
except ImportError as e:
    print(f"FAILED: Core Python dependency missing. {e}")
    sys.exit(1)

# 2. Verify R Environment and Dependencies
try:
    import rpy2.robjects as robjects
    from rpy2.robjects.packages import importr
    import rpy2.rinterface_lib.callbacks

    rpy2.rinterface_lib.callbacks.logger.setLevel(logging.ERROR)

    stream = importr("stream")
    streamMOA = importr("streamMOA")
    mlbench = importr("mlbench")
    print("OK: R environment and packages (stream, streamMOA, mlbench) loaded.")
except Exception as e:
    print(f"FAILED: R dependencies initialization error. {e}")
    sys.exit(1)

# 3. Verify Python Model Imports
try:
    from models.AutoCloud.AutoCloud import AutoCloud, AutoCloudParameterSearchSpace
    from models.CEDAS.CEDAS import CEDAS, CEDASParameterSearchSpace
    from models.CluStream.CluStream import CluStream, CluStreamParameterSearchSpace
    from models.DenStream.DenStream import DenStream, DenStreamParameterSearchSpace
    from models.MicroTEDAClus.MicroTEDAClus import (
        MicroTEDAClus,
        MicroTEDAClusParameterSearchSpace,
    )
    from models.StreamKM.StreamKM import StreamKM, StreamKMParameterSearchSpace
    from models.TEDACloud.TEDACloud import TEDACloud, TEDACloudParameterSearchSpace

    print("OK: All model and search space classes imported.")
except ImportError as e:
    print(f"FAILED: Model import error. {e}")
    sys.exit(1)


# 4. Verify Dataset Loading Mechanisms
def verify_datasets():
    # Synthetic
    try:
        X, y = make_blobs(n_samples=100, centers=3, n_features=2)[:2]
        assert X.shape == (100, 2)
    except Exception as e:
        print(f"FAILED: make_blobs failed. {e}")
        sys.exit(1)

    # Remote ARFF
    try:
        url = "https://raw.githubusercontent.com/deric/clustering-benchmark/master/src/main/resources/datasets/artificial/cluto-t7-10k.arff"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req) as response:
            data, _ = arff.loadarff(StringIO(response.read().decode("utf-8")))
        assert len(data) > 0
    except Exception as e:
        print(f"FAILED: Remote ARFF dataset fetch failed. {e}")
        sys.exit(1)

    # R Streams
    try:
        robjects.r("dsd <- DSD_Benchmark(1)")
        robjects.r("data <- get_points(dsd, n=10, info=TRUE)")
        X_r = np.array(robjects.r("as.matrix(data[, -ncol(data)])"))
        assert X_r.shape[0] == 10
    except Exception as e:
        print(f"FAILED: R stream generation failed. {e}")
        sys.exit(1)

    print("OK: Dataset retrieval and generation mechanisms validated.")


# 5. Verify Model API Constraints (Predict -> Fit -> Microcluster Count)
def verify_model_apis():
    dummy_X = np.random.rand(10, 2)

    # Map to instantiate with minimal required parameters per model_summary.md
    models_to_test = {
        "AutoCloud": AutoCloud(),
        "CEDAS": CEDAS(r0=1.0, decay_period=100, min_samples=2),
        "CluStream": CluStream(),
        "DenStream": DenStream(),
        "MicroTEDAClus": MicroTEDAClus(),
        "StreamKM": StreamKM(),
        "TEDACloud": TEDACloud(),
    }

    for name, model in models_to_test.items():
        try:
            preds = model.predict(dummy_X)
            assert len(preds) == len(dummy_X), f"{name} predict() dimension mismatch."

            model.fit(dummy_X)

            mc_count = model.microcluster_count()
            assert isinstance(
                mc_count, int
            ), f"{name} microcluster_count() must return int."
        except Exception as e:
            print(f"FAILED: {name} API validation failed.")
            traceback.print_exc()
            sys.exit(1)

    print("OK: Model API execution (predict, fit, microcluster_count) validated.")


# 6. Verify Optuna Search Spaces
def verify_search_spaces():
    spaces = [
        AutoCloudParameterSearchSpace(),
        CEDASParameterSearchSpace(),
        CluStreamParameterSearchSpace(),
        DenStreamParameterSearchSpace(),
        MicroTEDAClusParameterSearchSpace(),
        StreamKMParameterSearchSpace(),
        TEDACloudParameterSearchSpace(),
    ]

    study = optuna.create_study()
    trial = study.ask()

    for space in spaces:
        try:
            params = space.sample_parameters(trial)
            assert isinstance(params, dict)
        except Exception as e:
            print(f"FAILED: {space.__class__.__name__} failed to sample. {e}")
            sys.exit(1)

    print("OK: Optuna search spaces validated.")


if __name__ == "__main__":
    print("--- Starting Environment Verification ---")
    verify_datasets()
    verify_model_apis()
    verify_search_spaces()
    print("--- Environment Verification Complete. All checks passed. ---")
    sys.exit(0)
