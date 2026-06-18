import csv
import sys
from river.datasets.synth import RandomRBF


def generate_csv(filename: str, num_instances: int, num_features: int) -> None:
    stream = RandomRBF(n_classes=5, n_features=num_features, n_centroids=50)

    with open(filename, "w", newline="") as file:
        writer = csv.writer(file)
        for _, (x, _) in zip(range(num_instances), stream):
            writer.writerow([x[i] for i in range(num_features)])


if __name__ == "__main__":
    instances = int(sys.argv[1]) if len(sys.argv) > 1 else 100000
    features = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    generate_csv("stream_data.csv", instances, features)
