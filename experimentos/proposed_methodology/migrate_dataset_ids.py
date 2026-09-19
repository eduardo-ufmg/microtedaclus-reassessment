#!/usr/bin/env python3
"""Rename existing result files from legacy dataset IDs to canonical IDs.

This utility only renames files. It never imports or runs an experiment.
Use without arguments for a dry run; pass ``--apply`` to perform renames.
"""

import argparse
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
MIGRATION_DIRS = (
    SCRIPT_DIR / "results",
    SCRIPT_DIR / "results_params",
    SCRIPT_DIR.parent / "multiple_runs" / "results",
)
DATASET_ID_MAP = {
    "chameleon_ds7": "chameleon_ds3",
    "chameleon_ds8": "chameleon_ds4",
    "Benchmark(1)": "benchmark_1",
    "Benchmark(2)": "benchmark_2",
    "RandomRBFGeneratorEvents": "random_rbf_events",
    "STR-B1": "benchmark_1",
    "STR-B2": "benchmark_2",
}


def plan_renames():
    renames = []
    for directory in MIGRATION_DIRS:
        if not directory.is_dir():
            continue
        for path in sorted(directory.iterdir()):
            if not path.is_file():
                continue
            for old_id, new_id in DATASET_ID_MAP.items():
                prefix = f"{old_id}_"
                if path.name.startswith(prefix):
                    renames.append(
                        (path, path.with_name(new_id + path.name[len(old_id) :]))
                    )
                    break
    return renames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="perform the planned renames instead of printing a dry run",
    )
    args = parser.parse_args()

    renames = plan_renames()
    collisions = [destination for _, destination in renames if destination.exists()]
    if collisions:
        names = "\n".join(str(path) for path in collisions)
        raise SystemExit(f"Refusing to rename because destinations exist:\n{names}")

    action = "Renaming" if args.apply else "Would rename"
    for source, destination in renames:
        print(f"{action}: {source.name} -> {destination.name}")
        if args.apply:
            source.rename(destination)

    print(f"Planned renames: {len(renames)}")
    if not args.apply:
        print("Dry run only. Re-run with --apply to perform these file renames.")


if __name__ == "__main__":
    main()
