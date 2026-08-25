#!/usr/bin/env python3
import os
import sys

DIR_RESULTS = "./results"
DIR_PARAMS = "./results_params"

RUNS = 94

DATASETS = [
    "cassini",
    "chameleon_ds7",
    "chameleon_ds8",
    "RandomRBFGeneratorEvents",
    "Benchmark(1)",
    "Benchmark(2)",
    "SCL-1",
    "SCL-2",
    "SCL-4",
    "SCL-8",
    "SCL-16",
    "SCL-32",
    "SCL-64",
    "SCL-128",
]

MODELS = [
    "CluStream",
    "DenStream",
    "MicroTEDAClus",
    "CEDAS",
    "TEDACloud",
    "AutoCloud",
    "StreamKM",
]


def main():
    if not os.path.exists(DIR_RESULTS) or not os.path.exists(DIR_PARAMS):
        print("Error: Target directories do not exist.", file=sys.stderr)
        sys.exit(1)

    total_expected_per_dataset = len(MODELS) * RUNS

    complete_datasets = []
    incomplete_datasets = []

    for dataset in DATASETS:
        completed_runs = 0
        for model in MODELS:
            for run in range(RUNS):
                res_file = os.path.join(DIR_RESULTS, f"{dataset}_{model}_{run}.csv")
                param_file = os.path.join(DIR_PARAMS, f"{dataset}_{model}_{run}.csv")

                if os.path.exists(res_file) and os.path.exists(param_file):
                    completed_runs += 1

        if completed_runs == total_expected_per_dataset:
            complete_datasets.append(dataset)
        else:
            incomplete_datasets.append(
                (dataset, completed_runs, total_expected_per_dataset)
            )

    print(
        f"Total required (dataset, model, run) tuples per dataset: {total_expected_per_dataset}"
    )
    print("-" * 50)

    print("Complete Datasets:")
    if not complete_datasets:
        print("  <None>")
    else:
        for ds in complete_datasets:
            print(f"  [+] {ds}")

    print("-" * 50)
    print("Incomplete Datasets:")
    if not incomplete_datasets:
        print("  <None>")
    else:
        for ds, completed, total in incomplete_datasets:
            ratio = completed / total
            print(f"  [-] {ds}: {completed:4d} / {total:4d} ({ratio * 100:.2f}%)")


if __name__ == "__main__":
    main()
