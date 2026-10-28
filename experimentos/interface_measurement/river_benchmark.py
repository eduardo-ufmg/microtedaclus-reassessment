import sys
import time

from memory_profiler import memory_usage
from river import cluster


def make_clusterer(algorithm):
    if algorithm == "CluStream":
        # MOA WithKmeans: maxNumKernels=100, k=5. time_window/time_gap have no
        # MOA counterpart (MOA has no offline phase); left at river defaults.
        return cluster.CluStream(n_macro_clusters=5, max_micro_clusters=100)
    elif algorithm == "DenStream":
        # MOA: epsilon=0.02, mu=1.0, beta=0.2, lambda=0.001, initPoints=100,
        # offline=2.0. River's `mu` plays the role of MOA's (mu * offline)
        # threshold directly (no separate offline multiplier exists), so
        # mu is set to 1.0 * 2.0 = 2.0 here. lambda -> decaying_factor,
        # initPoints -> n_samples_init.
        return cluster.DenStream(
            decaying_factor=0.001,
            beta=0.6,
            mu=2.0,
            epsilon=0.02,
            n_samples_init=100,
        )
    elif algorithm == "StreamKM":
        # MOA StreamKM: sizeCoreset=10000, numClusters=5, length=100000.
        # River's STREAMKMeans has no coreset-size or stream-length
        # parameter; chunk_size (buffer flushed into an incremental KMeans)
        # is the closest analogue to sizeCoreset and is set accordingly.
        # `length` has no equivalent and is dropped.
        return cluster.STREAMKMeans(chunk_size=10000, n_clusters=5)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def run_river_experiment(csv_path, algorithm):
    clusterer = make_clusterer(algorithm)

    start_time = time.perf_counter()

    with open(csv_path, "r") as f:
        for line in f:
            values = [float(x) for x in line.strip().split(",")]
            x = dict(enumerate(values))
            clusterer.learn_one(x)

    end_time = time.perf_counter()
    print(f"River Execution Time [{algorithm}]: {end_time - start_time:.4f} seconds")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(
            "Usage: python3 river_benchmark.py <csv_file> <Algorithm: CluStream|DenStream|StreamKM>"
        )
        sys.exit(1)

    csv_file = sys.argv[1]
    algorithm = sys.argv[2]

    mem_use = memory_usage((run_river_experiment, (csv_file, algorithm)), interval=0.1)
    print(f"Peak Memory Usage (MB) [{algorithm}]: {max(mem_use):.2f}")
