#!/bin/bash
# The pilot of the modernized experiment: every model selects its hyperparameters by the
# rules, real2, on the runs 94 to 125 of the synthetic datasets, which the experiment does
# not use, and nothing is evaluated. `python -m benchmark.range_check results_pilot` then
# checks the ranges. It continues from the tasks already written.
# Options: --workers N and --memory-limit GIGABYTES (default 16 and 5); --test runs only the
# first run; --trials N, --root DIR and --datasets D ... serve a test at small scale.
set -e
cd "$(dirname "$0")"

workers=16
memory_limit=5
runs=(94 125)
trials=100
root=.
models=(MicroTEDAClusPlus DenStream DBSTREAM BIRCH MiniBatchKMeans BICO)
datasets=(benchmark_1 benchmark_2 random_rbf_events cassini chameleon_ds3 chameleon_ds4
    SCL-1 SCL-2 SCL-4 SCL-8 SCL-16 SCL-32 SCL-64)
while [ $# -gt 0 ]; do
    case "$1" in
        --workers) workers=$2; shift 2 ;;
        --memory-limit) memory_limit=$2; shift 2 ;;
        --test) runs=(94 94); shift ;;
        --trials) trials=$2; shift 2 ;;
        --root) root=$2; shift 2 ;;
        --datasets) datasets=(); shift; while [ $# -gt 0 ] && [[ "$1" != --* ]]; do
            datasets+=("$1"); shift; done ;;
        *) echo "ERROR: unknown option $1" >&2; exit 2 ;;
    esac
done

.venv/bin/python -m benchmark.runner "$root/results_pilot" --protocol real2 --tune-only \
    --length-factor 5 --runs "${runs[@]}" --datasets "${datasets[@]}" \
    --models "${models[@]}" --trials "$trials" --workers "$workers" \
    --memory-limit "$memory_limit" --stall-limit 10800
echo "The pilot is complete."
