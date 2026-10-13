#!/bin/bash
# The experiment of the chapter, in three parts that continue from what is already written:
# the models that fit the time budget, MicroTEDAClus+ under the previous protocol, and the
# model whose search is the most expensive. Arguments go to every part, e.g. --workers 10.
set -e
cd "$(dirname "$0")"
.venv/bin/python -m benchmark.runner results_main "$@" \
    --models MicroTEDAClusPlus CluStream DenStream DBSTREAM BIRCH MiniBatchKMeans BICO
.venv/bin/python -m benchmark.runner results_previous_protocol "$@" \
    --protocol previous --models MicroTEDAClusPlus
.venv/bin/python -m benchmark.runner results_main "$@" --models MCMSTStream
