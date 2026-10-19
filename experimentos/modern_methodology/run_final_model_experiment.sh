#!/bin/bash
# The final version of MicroTEDAClus+ on the real datasets, in three parts that continue from
# what is already written:
#   1. its quality, tuned by the widened rules and evaluated on the 32 first runs, as every
#      model of the real-data experiment, in results_final_model;
#   2. the execution time of it and of the benchmark models of the real-data experiment, with
#      their selected hyperparameters, on the ten first runs, one task at a time, in
#      results_timing_real;
#   3. their memory, on the same tasks, in results_memory_real.
# Options: --workers N and --memory-limit GIGABYTES for the parallel parts, 1 and 3 (default
# 16 and 5); --test runs only the first run of every part, for a test at small scale.
set -e
cd "$(dirname "$0")"

workers=16
memory_limit=5
runs=(0 31)
cost_runs=(0 9)
while [ $# -gt 0 ]; do
    case "$1" in
        --workers) workers=$2; shift 2 ;;
        --memory-limit) memory_limit=$2; shift 2 ;;
        --test) runs=(0 0); cost_runs=(0 0); shift ;;
        *) echo "ERROR: unknown option $1" >&2; exit 2 ;;
    esac
done

benchmarks=(DenStream DBSTREAM BIRCH MiniBatchKMeans BICO)
parallel=(--workers "$workers" --memory-limit "$memory_limit" --stall-limit 10800)

.venv/bin/python -m benchmark.runner results_final_model --protocol widened \
    --runs "${runs[@]}" \
    --datasets insects_abrupt insects_incremental insects_incremental_reoccurring gas_drift \
    kddcup99 covertype pendigits shuttle \
    --models MicroTEDAClusPlusFinal "${parallel[@]}"
echo "The quality part is complete."

.venv/bin/python -m benchmark.timing --results results_real --out results_timing_real \
    --runs "${cost_runs[@]}" --models "${benchmarks[@]}"
.venv/bin/python -m benchmark.timing --results results_final_model --out results_timing_real \
    --runs "${cost_runs[@]}"
echo "The time part is complete."

.venv/bin/python -m benchmark.memory --results results_real --out results_memory_real \
    --runs "${cost_runs[@]}" --models "${benchmarks[@]}" "${parallel[@]}"
.venv/bin/python -m benchmark.memory --results results_final_model --out results_memory_real \
    --runs "${cost_runs[@]}" "${parallel[@]}"
echo "The experiment is complete."
