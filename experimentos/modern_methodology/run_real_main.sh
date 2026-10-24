#!/bin/bash
# The real-data experiment, in three parts that continue from what is already written:
#   1. the quality of every model, tuned by the real-data rules, on the runs 126 to 219, which
#      no earlier analysis of the real datasets used, in results_real_main;
#   2. their execution time on the runs 126 to 157, one task at a time, in
#      results_timing_real_main; the pass fails if it takes more than 24 hours;
#   3. their memory, on the same tasks, in results_memory_real_main.
# Options: --workers N and --memory-limit GIGABYTES for the parallel parts, 1 and 3 (default
# 16 and 5); --test runs only the first run of every part; --trials N, --root DIR and
# --datasets D ... serve a test at small scale.
set -e
cd "$(dirname "$0")"

workers=16
memory_limit=5
runs=(126 219)
cost_runs=(126 157)
trials=100
root=.
datasets=(insects_abrupt insects_incremental insects_incremental_reoccurring gas_drift kddcup99
    covertype pendigits shuttle)
while [ $# -gt 0 ]; do
    case "$1" in
        --workers) workers=$2; shift 2 ;;
        --memory-limit) memory_limit=$2; shift 2 ;;
        --test) runs=(126 126); cost_runs=(126 126); shift ;;
        --trials) trials=$2; shift 2 ;;
        --root) root=$2; shift 2 ;;
        --datasets) datasets=(); shift; while [ $# -gt 0 ] && [[ "$1" != --* ]]; do
            datasets+=("$1"); shift; done ;;
        *) echo "ERROR: unknown option $1" >&2; exit 2 ;;
    esac
done

models=(MicroTEDAClusPlusNoNoise DenStream DBSTREAM BIRCH MiniBatchKMeans BICO)
parallel=(--workers "$workers" --memory-limit "$memory_limit" --stall-limit 10800)

.venv/bin/python -m benchmark.runner "$root/results_real_main" --protocol real \
    --runs "${runs[@]}" --datasets "${datasets[@]}" --models "${models[@]}" \
    --trials "$trials" "${parallel[@]}"
echo "The quality part is complete."

timeout 24h .venv/bin/python -m benchmark.timing --results "$root/results_real_main" \
    --out "$root/results_timing_real_main" --runs "${cost_runs[@]}" --models "${models[@]}"
echo "The time part is complete."

.venv/bin/python -m benchmark.memory --results "$root/results_real_main" \
    --out "$root/results_memory_real_main" --runs "${cost_runs[@]}" --models "${models[@]}" \
    "${parallel[@]}"
echo "The experiment is complete."
