#!/bin/bash
# The experiment on the real datasets: every model, tuned by the widened rules, on their 32
# first runs. It continues from the tasks already written in results_real. Arguments go to
# the runner, e.g. --workers 12 --memory-limit 4.
set -e
cd "$(dirname "$0")"
.venv/bin/python -m benchmark.runner results_real --protocol widened --runs 0 31 \
    --datasets insects_abrupt insects_incremental insects_incremental_reoccurring gas_drift \
    kddcup99 covertype pendigits shuttle \
    --stall-limit 21600 "$@"
echo "The experiment is complete."
