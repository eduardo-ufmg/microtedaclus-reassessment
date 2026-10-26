#!/bin/bash
# The modernized experiment, in four parts that continue from what is already written:
#   1. the quality of every model, tuned by the rules, real2, on the runs 126 to 219 of the
#      synthetic datasets, which no earlier analysis used, in results_main;
#   2. MicroTEDAClus+ tuned by the previous protocol on the same streams, in
#      results_previous_protocol;
#   3. the execution time of the models of part 1 on the runs 126 to 157, one task at a
#      time, in results_timing; the pass fails if it takes more than 24 hours;
#   4. their memory, on the same tasks, in results_memory.
# Options: --workers N and --memory-limit GIGABYTES for the parallel parts, 1, 2 and 4
# (default 16 and 5); --test runs only the first run of every part; --trials N, --root DIR
# and --datasets D ... serve a test at small scale.
set -e
cd "$(dirname "$0")"

workers=16
memory_limit=5
runs=(126 219)
cost_runs=(126 157)
trials=100
root=.
datasets=(benchmark_1 benchmark_2 random_rbf_events cassini chameleon_ds3 chameleon_ds4
    SCL-1 SCL-2 SCL-4 SCL-8 SCL-16 SCL-32 SCL-64)
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

models=(MicroTEDAClusPlus DenStream DBSTREAM BIRCH MiniBatchKMeans BICO)
parallel=(--workers "$workers" --memory-limit "$memory_limit" --stall-limit 10800)

.venv/bin/python -m benchmark.runner "$root/results_main" --protocol real2 --length-factor 5 \
    --runs "${runs[@]}" --datasets "${datasets[@]}" --models "${models[@]}" \
    --trials "$trials" "${parallel[@]}"
echo "The quality part is complete."

.venv/bin/python -m benchmark.runner "$root/results_previous_protocol" --protocol previous \
    --length-factor 5 --runs "${runs[@]}" --datasets "${datasets[@]}" \
    --models MicroTEDAClusPlus --trials "$trials" "${parallel[@]}"
echo "The part of the previous protocol is complete."

timeout 24h .venv/bin/python -m benchmark.timing --results "$root/results_main" \
    --out "$root/results_timing" --runs "${cost_runs[@]}" --models "${models[@]}"
echo "The time part is complete."

.venv/bin/python -m benchmark.memory --results "$root/results_main" \
    --out "$root/results_memory" --runs "${cost_runs[@]}" --models "${models[@]}" \
    "${parallel[@]}"
echo "The experiment is complete."
