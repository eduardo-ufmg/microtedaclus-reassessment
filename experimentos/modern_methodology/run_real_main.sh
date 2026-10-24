#!/bin/bash
# The real-data experiment, in three parts that continue from what is already written:
#   1. the quality of every model, tuned by the real-data rules, on the runs 0 to 93, in
#      results_real_main;
#   2. their execution time on the runs 0 to 31, one task at a time, in
#      results_timing_real_main;
#   3. their memory, on the same tasks, in results_memory_real_main.
# Options: --workers N and --memory-limit GIGABYTES for the parallel parts, 1 and 3 (default
# 16 and 5); --test runs only the first run of every part; --trials N and --root DIR serve a
# test at small scale.
set -e
cd "$(dirname "$0")"

workers=16
memory_limit=5
runs=(0 93)
cost_runs=(0 31)
trials=100
root=.
while [ $# -gt 0 ]; do
    case "$1" in
        --workers) workers=$2; shift 2 ;;
        --memory-limit) memory_limit=$2; shift 2 ;;
        --test) runs=(0 0); cost_runs=(0 0); shift ;;
        --trials) trials=$2; shift 2 ;;
        --root) root=$2; shift 2 ;;
        *) echo "ERROR: unknown option $1" >&2; exit 2 ;;
    esac
done

datasets=(insects_abrupt insects_incremental insects_incremental_reoccurring gas_drift kddcup99
    covertype pendigits shuttle)
models=(MicroTEDAClusPlusNoNoise DenStream DBSTREAM BIRCH MiniBatchKMeans BICO)
parallel=(--workers "$workers" --memory-limit "$memory_limit" --stall-limit 10800)

.venv/bin/python -m benchmark.runner "$root/results_real_main" --protocol real \
    --runs "${runs[@]}" --datasets "${datasets[@]}" --models "${models[@]}" \
    --trials "$trials" "${parallel[@]}"
echo "The quality part is complete."

.venv/bin/python -m benchmark.timing --results "$root/results_real_main" \
    --out "$root/results_timing_real_main" --runs "${cost_runs[@]}" --models "${models[@]}"
echo "The time part is complete."

.venv/bin/python -m benchmark.memory --results "$root/results_real_main" \
    --out "$root/results_memory_real_main" --runs "${cost_runs[@]}" --models "${models[@]}" \
    "${parallel[@]}"
echo "The experiment is complete."
