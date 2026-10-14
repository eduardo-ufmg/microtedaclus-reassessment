"""Discard the results of a model, so that the experiment runs it again.

For when the definition of a model changes, such as the hyperparameters that
its search selects. The tasks of the model leave every part of the results, and
the next call of the experiment runs them as tasks that are missing. No part of
the experiment may be running.

Usage (from experimentos/modern_methodology):
    python -m benchmark.discard MODEL
"""

import argparse
import shutil

import pandas as pd

from benchmark import analysis
from benchmark.models import MODELS

PARTS = (analysis.MAIN, analysis.PREVIOUS_PROTOCOL, analysis.TIMING, analysis.MEMORY)


def discard(model: str) -> None:
    for part in PARTS:
        tasks_file = part / "tasks.csv"
        if tasks_file.exists():
            tasks = pd.read_csv(tasks_file, dtype=str, keep_default_na=False)
            kept = tasks[tasks.model != model]
            partial = tasks_file.with_suffix(".partial")
            kept.to_csv(partial, index=False)
            partial.replace(tasks_file)
            print(
                f"{part.name}: {len(tasks) - len(kept)} tasks of {len(tasks)} discarded"
            )
        folders = sorted((part / "batches").glob(f"*/{model}"))
        for folder in folders:
            shutil.rmtree(folder)
        if folders:
            print(f"{part.name}: the batches of {len(folders)} datasets discarded")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("model", choices=list(MODELS))
    discard(parser.parse_args().model)


if __name__ == "__main__":
    main()
