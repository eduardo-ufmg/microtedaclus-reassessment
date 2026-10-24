#!/bin/bash
# The pilot of the real-data experiment: every model selects its hyperparameters by the
# real-data rules on the runs 94 to 125, which the experiment does not use, and nothing is
# evaluated. `python -m benchmark.range_check results_real_pilot` then checks the ranges.
# It continues from the tasks already written.
# Options: --workers N and --memory-limit GIGABYTES (default 16 and 5); --models M ... to
# select the models again in a new directory, --out DIR (default results_real_pilot);
# --test runs only the first run; --trials N and --root DIR serve a test at small scale.
set -e
cd "$(dirname "$0")"

workers=16
memory_limit=5
runs=(94 125)
trials=100
root=.
out=results_real_pilot
models=(MicroTEDAClusPlus DenStream DBSTREAM BIRCH MiniBatchKMeans BICO)
while [ $# -gt 0 ]; do
    case "$1" in
        --workers) workers=$2; shift 2 ;;
        --memory-limit) memory_limit=$2; shift 2 ;;
        --test) runs=(94 94); shift ;;
        --trials) trials=$2; shift 2 ;;
        --root) root=$2; shift 2 ;;
        --out) out=$2; shift 2 ;;
        --models) models=(); shift; while [ $# -gt 0 ] && [[ "$1" != --* ]]; do
            models+=("$1"); shift; done ;;
        *) echo "ERROR: unknown option $1" >&2; exit 2 ;;
    esac
done

datasets=(insects_abrupt insects_incremental insects_incremental_reoccurring gas_drift kddcup99
    covertype pendigits shuttle)

.venv/bin/python -m benchmark.runner "$root/$out" --protocol real --tune-only \
    --runs "${runs[@]}" --datasets "${datasets[@]}" --models "${models[@]}" \
    --trials "$trials" --workers "$workers" --memory-limit "$memory_limit" --stall-limit 10800
echo "The pilot is complete."
