# Reference material (deprecated)

This directory keeps the work that preceded `\chapter{Model Improvement}`. Nothing here is
compiled into the thesis or imported by the live code. It mirrors the original paths, so
`referencia/<path>` used to live at `<path>`.

Everything here was last in place, and runnable, at the tag `pre-model-improvement`. To run
any of it, check that tag out; the scripts are not expected to work from this directory.

## Contents

| Path | What it is |
|---|---|
| `monografia/CorrectedModel/` | Former chapter "Model Correction": stage-by-stage corrections (order-dependent density, dimensionality penalty, typicality, decay, macrocluster topology) and their tables. |
| `monografia/ImprovedModel/` | Former chapter "Improved Model" (base improvement and `m`-function study). |
| `experimentos/proposed_methodology/experiment_runner.py`, `analysis_pipeline.py`, `stages_config.py` | The iterative ("stage") experiment pipeline behind those chapters. |
| `experimentos/proposed_methodology/profile_base_improvements.py` | Profiling suite of the former base improvement. |
| `experimentos/proposed_methodology/models/MicroTEDAClusCorrected/` | The stage models and the decay and macrocluster-topology investigations. |
| `experimentos/proposed_methodology/investigation/*.csv` | Summaries of the exploratory investigation that led to the new chapter. |

## Why it was replaced

The iterative pipeline did not follow the methodology chapter: it used 32 runs and
64 tuning trials (instead of 94 and 100), summarized quality by the mean ARI and the mean of
the last 20% of batches (instead of the median and the penultimate batch), defined the ARI
drop as max - min, and measured CPU instead of wall-clock time. Its tables are therefore not
comparable with the main experiment's.

## Local-only data

The raw outputs (`results*`, `metrics*`, `statistical_analysis*`, `plots/`) are not tracked.
They exist only on the machine where they were produced:

- `*_experiments/`: outputs of the iterative pipeline.
- `results_investigation/`: per-batch results of the exploratory investigation
  (32 runs, 64 trials), superseded by `experimentos/proposed_methodology/results_model_improvement/`.
