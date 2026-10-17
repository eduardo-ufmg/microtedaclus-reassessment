#!/bin/bash
# Confirmation of the improvement on the real datasets: the final model of the improvement
# rounds, then the two intermediate steps of its chain, tuned and evaluated on runs 0-31 of the
# eight real datasets, the runs of the real-data experiment, to be compared with its benchmark
# models and MicroTEDAClus+. It continues from the tasks already written in
# results_confirmation. Arguments go to the runner, e.g. --workers 16 --memory-limit 5.
set -e
cd "$(dirname "$0")"
for candidate in no_noise_link low_variance no_noise; do
    .venv/bin/python -m benchmark.improvement.run "$candidate" --runs 0 31 \
        --datasets insects_abrupt insects_incremental insects_incremental_reoccurring gas_drift \
        kddcup99 covertype pendigits shuttle \
        --out results_confirmation --stall-limit 10800 "$@"
done
echo "The confirmation is complete."
