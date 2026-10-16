# Improvement on the real datasets: log

The investigation of 2026-10-16 onward, on the branch `real-improvement`: the correction and
improvement effort of the Model Improvement chapter, repeated on six real datasets (INSECTS
abrupt, incremental and reoccurring, Gas, Pendigits, Shuttle), locally, by rounds. Code in
`benchmark/improvement/`; results in `results_improvement/` (git-ignored). Each entry is
written before its run and completed after it.

## Setup

- Development runs 32-63; protocol of the real-data experiment (widened rules, 100 trials,
  tuning segment of at most 10,000 samples, summary limit, standardization). Every candidate
  of a task uses the seeds of MicroTEDAClus+ in that task.
- Control: Chapter 3's final model on the copied engine. Replayed with the hyperparameters
  of ten real-data tasks, it gives the same batch ARIs and summary sizes as MicroTEDAClus+
  (identical, not only close).
- Comparison (`python -m benchmark.improvement.compare CANDIDATE`): paired Wilcoxon with
  Holm over the datasets of a metric and Cliff's delta; outcome on the Overall ARI (six
  datasets) and the Steady-State ARI (all but INSECTS abrupt and reoccurring).
- Decision: keep if +; keep an analytically preferred candidate if = and its summary is not
  clearly larger; keep a ± one if, after diagnosing its losses, its gains outnumber them
  and no loss comes from a defect. A kept candidate becomes the control.

## Diagnosis (Phase 2, no new runs)

`python -m benchmark.improvement.diagnose`: the selected configurations of MicroTEDAClus+
in runs 0-3 of the real-data experiment, replayed on their evaluation segments, against
Mini-batch k-means replayed the same way. Medians over the batches, then over the runs:

| | INS-A | INS-I | INS-R | GAS | PEN | SHU |
|---|---|---|---|---|---|---|
| ARI, MicroTEDAClus+ / k-means | 0.03 / 0.18 | 0.08 / 0.17 | 0.01 / 0.13 | 0.16 / 0.22 | 0.42 / 0.50 | 0.50 / 0.40 |
| share labelled noise | 0.46 | 0.09 | 0.18 | 0.04 | 0.01 | 0.02 |
| predicted clusters (classes in a batch) | 4.5 (5) | 7.5 (6) | 1 (5) | 8 (6) | 8.5 (10) | 12 (3) |
| active / all microclusters | 10 / 94 | 8 / 95 | 3.5 / 47 | 28.5 / 92 | 12 / 78 | 196 / 608 |
| mature (three samples or more) | 9 | 9 | 4 | 31 | 16 | 288 |

Findings:

1. The model is poor on INSECTS already on the tuning segment, where its configuration was
   selected (ARI 0.01 to 0.11 in runs 0-1): drift is not the main cause.
2. It does not fragment the classes: it predicts about as many clusters as there are
   classes. On INSECTS reoccurring it predicts a single cluster.
3. The intersection graph is not the cause: among the mature microclusters of INSECTS
   there are 6 to 7 components and the mean degree is below one.
4. The cause is upstream: about nine microclusters in ten never reach three samples, so
   they never mature and never become active. The activation, the macroclusters, the
   prediction and the noise boundary all rest on a handful of mature microclusters, and
   the samples far from them are labelled as noise.
5. The selected variance limits sit at the low end of the range (0.04 to 0.06 of the
   stream's variance), as the ends table of the real-data experiment showed, and the
   selected decays are often short (10 to 14 samples in some runs).

Hypothesis: young microclusters reject most of the samples of their own class. A sample
joins a microcluster when its standardized squared distance q stays within m^2, and m(k) is
close to 1 for the first tens of samples. In 33 dimensions the standardized distance of a
sample of the same distribution concentrates around 1, so a strict threshold rejects a large
share of them; before three samples, the variance gate decides alone.

Consequence for the order of the rounds: the candidates that act on the formation of
microclusters, the m variants and the dimension-aware threshold (Chapter 3's chi-threshold,
which collapsed on 2-D shapes but is motivated by the dimension), go before the geometry and
the density, which act on the shape of the test and on the activation. The geometry still
follows: with a covariance, q is a Mahalanobis distance and concentrates in the same way.

## Round 1: variance range

- `low_variance`: the variance limit searched from a tenth of the lower end of the
  relative-variance range. Reason: that end binds in 56% to 84% of the runs of INSECTS,
  Covertype and Pendigits in the real-data experiment.
