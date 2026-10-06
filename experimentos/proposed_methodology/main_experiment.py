import itertools
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

import protocol
from editorial_config import DATASET_ORDER
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
from streams import get_dataset, r_session

DIR_RESULTS = "./results"
DIR_PARAMS = "./results_params"
DIR_SAMPLES = "./data_samples"
os.makedirs(DIR_RESULTS, exist_ok=True)
os.makedirs(DIR_PARAMS, exist_ok=True)
os.makedirs(DIR_SAMPLES, exist_ok=True)

# -----------------------------------------------------------------------------
# Configuration (the tuning and evaluation protocol lives in protocol.py)
# -----------------------------------------------------------------------------
RUNS = 94  # Aligned with k* = 94

DATASETS = [
    *DATASET_ORDER,
    "SCL-1",
    "SCL-2",
    "SCL-4",
    "SCL-8",
    "SCL-16",
    "SCL-32",
    "SCL-64",
]

MODEL_MAP = {
    "CluStream": (CluStream, CluStreamParameterSearchSpace),
    "DenStream": (DenStream, DenStreamParameterSearchSpace),
    "MicroTEDAClus": (MicroTEDAClus, MicroTEDAClusParameterSearchSpace),
    "CEDAS": (CEDAS, CEDASParameterSearchSpace),
    "TEDACloud": (TEDACloud, TEDACloudParameterSearchSpace),
    "AutoCloud": (AutoCloud, AutoCloudParameterSearchSpace),
    "StreamKM": (StreamKM, StreamKMParameterSearchSpace),
}


def save_data_sample(dataset, run, X_eval, y_eval, split_idx, seed):
    """Isolate Data Sampling (10% per batch constraint) per dataset per run."""
    sample_file_path = os.path.join(DIR_SAMPLES, f"{dataset}_{run}.csv")
    if os.path.exists(sample_file_path):
        return

    rng = np.random.default_rng(seed)
    sample_indices = []
    for b in range(len(X_eval) // protocol.BATCH_SIZE):
        start_idx = b * protocol.BATCH_SIZE
        sample_size = max(1, protocol.BATCH_SIZE // 10)
        idx = rng.choice(protocol.BATCH_SIZE, size=sample_size, replace=False)
        sample_indices.extend(start_idx + idx)

    sample_indices = np.sort(sample_indices)
    global_indices = sample_indices + split_idx
    sampled_data = np.column_stack(
        (global_indices, X_eval[sample_indices], y_eval[sample_indices])
    )
    cols = ["sample_idx"] + [f"x{i}" for i in range(X_eval.shape[1])] + ["y"]
    pd.DataFrame(sampled_data, columns=cols).to_csv(sample_file_path, index=False)


def main():
    r_session()  # fail early if the R dependencies are missing

    # Flatten loop for atomic progress tracking
    tasks = list(itertools.product(DATASETS, MODEL_MAP.keys(), range(RUNS)))

    with tqdm(total=len(tasks), desc="Experiments") as pbar:
        for dataset, model_name, run in tasks:
            output_file = os.path.join(DIR_RESULTS, f"{dataset}_{model_name}_{run}.csv")
            params_file = os.path.join(DIR_PARAMS, f"{dataset}_{model_name}_{run}.csv")

            # Persistency constraint check
            if os.path.exists(output_file) and os.path.exists(params_file):
                pbar.update(1)
                continue

            seed = protocol.run_seed(dataset, run)
            X, y = get_dataset(dataset, seed)
            split_idx = int(protocol.SPLIT_FRACTION * len(X))
            save_data_sample(
                dataset, run, X[split_idx:], y[split_idx:], split_idx, seed
            )

            model_class, search_space_class = MODEL_MAP[model_name]
            protocol.run_task(
                dataset,
                model_name,
                run,
                model_class,
                search_space_class,
                X,
                y,
                output_file,
                params_file,
            )
            pbar.update(1)


if __name__ == "__main__":
    main()
