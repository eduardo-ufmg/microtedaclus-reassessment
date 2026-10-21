#!/bin/bash
# The comparison of the final version of MicroTEDAClus+ with the benchmark models, extended in
# three parts that continue from what is already written:
#   1. the quality of the final version and of the benchmark models on the runs 64 to 125,
#      which with the runs 0 to 31 make 94 and leave out the development runs 32 to 63, in
#      results_final_model and results_real;
#   2. the execution time of the six models on the runs 10 to 31, which with the runs 0 to 9
#      make 32, one task at a time, in results_timing_real;
#   3. their memory, on the same tasks, in results_memory_real.
# Options: --workers N and --memory-limit GIGABYTES for the parallel parts, 1 and 3 (default
# 16 and 5); --test runs only the first run of every part, for a test at small scale.
set -e
cd "$(dirname "$0")"

workers=16
memory_limit=5
runs=(64 125)
cost_runs=(10 31)
while [ $# -gt 0 ]; do
    case "$1" in
        --workers) workers=$2; shift 2 ;;
        --memory-limit) memory_limit=$2; shift 2 ;;
        --test) runs=(64 64); cost_runs=(10 10); shift ;;
        *) echo "ERROR: unknown option $1" >&2; exit 2 ;;
    esac
done

datasets=(insects_abrupt insects_incremental insects_incremental_reoccurring gas_drift kddcup99
    covertype pendigits shuttle)
benchmarks=(DenStream DBSTREAM BIRCH MiniBatchKMeans BICO)
parallel=(--workers "$workers" --memory-limit "$memory_limit" --stall-limit 10800)

.venv/bin/python -m benchmark.runner results_final_model --protocol widened \
    --runs "${runs[@]}" --datasets "${datasets[@]}" --models MicroTEDAClusPlusFinal \
    "${parallel[@]}"
.venv/bin/python -m benchmark.runner results_real --protocol widened \
    --runs "${runs[@]}" --datasets "${datasets[@]}" --models "${benchmarks[@]}" \
    "${parallel[@]}"
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
