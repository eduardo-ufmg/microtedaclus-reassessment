#!/bin/bash
# The experiment of the chapter, in four parts that continue from what is already written:
# every model under the protocol by rules, MicroTEDAClus+ under the previous protocol, the
# execution time of the selected models, measured one task at a time, and their memory.
# Arguments go to the two first parts, e.g. --workers 16 --length-factor 5.
set -e
cd "$(dirname "$0")"
.venv/bin/python -m benchmark.runner results_main --models MicroTEDAClusPlus DenStream DBSTREAM BIRCH MiniBatchKMeans BICO "$@"
.venv/bin/python -m benchmark.runner results_previous_protocol "$@" \
    --protocol previous --models MicroTEDAClusPlus
.venv/bin/python -m benchmark.timing
.venv/bin/python -m benchmark.memory
echo "The experiment is complete."
