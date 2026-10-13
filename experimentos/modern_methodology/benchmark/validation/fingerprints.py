"""Fingerprints of the streams: does this machine generate the streams of the experiment?

A fingerprint is a small summary of a stream: its shape, the sums of its
coordinates and of their squares, and the number of samples of each label. The
fingerprints of the first runs of every dataset are stored with the code, and
this check compares them with what the machine generates, within the precision
of floating-point arithmetic.

Usage (from experimentos/modern_methodology):
    python -m benchmark.validation.fingerprints            # check
    python -m benchmark.validation.fingerprints --write    # store the fingerprints of this machine
"""

import json
import sys
from pathlib import Path

import numpy as np

from benchmark.datasets import DATASETS, load_stream

FILE = Path(__file__).with_name("fingerprints.json")
RUNS = range(3)


def fingerprint(X, y) -> dict:
    labels, counts = np.unique(y, return_counts=True)
    return {
        "shape": list(X.shape),
        "sums": X.sum(axis=0).tolist(),
        "sum_of_squares": float((X**2).sum()),
        "first_sample": X[0].tolist(),
        "label_counts": {
            str(label): int(count) for label, count in zip(labels, counts)
        },
    }


def matches(stored: dict, current: dict) -> bool:
    return (
        stored["shape"] == current["shape"]
        and stored["label_counts"] == current["label_counts"]
        and np.allclose(stored["sums"], current["sums"], rtol=1e-9, atol=1e-9)
        and np.allclose(
            stored["first_sample"], current["first_sample"], rtol=1e-9, atol=1e-12
        )
        and np.isclose(stored["sum_of_squares"], current["sum_of_squares"], rtol=1e-9)
    )


def current_fingerprints() -> dict:
    return {
        f"{dataset}/{run}": fingerprint(*load_stream(dataset, run))
        for dataset in DATASETS
        for run in RUNS
    }


def main():
    current = current_fingerprints()
    if "--write" in sys.argv:
        FILE.write_text(json.dumps(current, indent=1) + "\n")
        print(f"{len(current)} fingerprints stored")
        return
    stored = json.loads(FILE.read_text())
    different = [key for key in stored if not matches(stored[key], current[key])]
    if different:
        raise SystemExit(
            f"streams differ from the stored fingerprints: {', '.join(different)}"
        )
    print(f"{len(stored)} streams match the stored fingerprints")


if __name__ == "__main__":
    main()
