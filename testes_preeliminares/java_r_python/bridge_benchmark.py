import logging
import sys
import time

import rpy2.rinterface_lib.callbacks
import rpy2.robjects.packages as rpackages
from memory_profiler import memory_usage
from rpy2.robjects import FloatVector, r

rpy2.rinterface_lib.callbacks.logger.setLevel(logging.ERROR)

stream = rpackages.importr("stream")
streamMOA = rpackages.importr("streamMOA")
r_update = r["update"]


def run_bridge_experiment(csv_path, algorithm):
    if algorithm == "CluStream":
        clusterer = streamMOA.DSC_CluStream(m=100, k=5)
    elif algorithm == "DenStream":
        kwargs_denstream = {
            "epsilon": 0.02,
            "mu": 1.0,
            "beta": 0.2,
            "lambda": 0.001,
            "initPoints": 100,
            "offline": 2.0,
        }
        clusterer = streamMOA.DSC_DenStream(**kwargs_denstream)
    elif algorithm == "StreamKM":
        clusterer = streamMOA.DSC_StreamKM(
            sizeCoreset=10000, numClusters=5, length=100000
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    start_time = time.perf_counter()

    with open(csv_path, "r") as f:
        for line in f:
            values = [float(x) for x in line.strip().split(",")]
            r_instance = r.matrix(FloatVector(values), nrow=1)
            r_update(clusterer, r_instance, n=1)

    end_time = time.perf_counter()
    print(f"Bridge Execution Time [{algorithm}]: {end_time - start_time:.4f} seconds")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(
            "Usage: python3 bridge_benchmark.py <csv_file> <Algorithm: CluStream|DenStream|StreamKM>"
        )
        sys.exit(1)

    csv_file = sys.argv[1]
    algorithm = sys.argv[2]

    mem_use = memory_usage((run_bridge_experiment, (csv_file, algorithm)), interval=0.1)
    print(f"Peak Memory Usage (MB) [{algorithm}]: {max(mem_use):.2f}")
