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

Result (192 tasks each; control all OK, low_variance 3 failed by the summary limit):

| | INS-A | INS-I | INS-R | GAS | PEN | SHU |
|---|---|---|---|---|---|---|
| Overall ARI, median | 0.011 > **0.142** | 0.087 > **0.137** | 0.013 > **0.107** | 0.163 > 0.154 | 0.406 > **0.505** | 0.458 > 0.417 |
| Steady-State ARI | (not deciding) | 0.105 > 0.164 | (not deciding) | 0.093 > 0.076 | 0.372 > **0.543** | 0.415 > 0.365 |
| steady-state summary | 77 > 451 | 93 > 106 | 9 > 215 | 112 > 122 | 71 > 263 | 193 > 161 |

Outcome **+** (5 significant gains, no loss). The summaries are larger where the model
improves; the variance limit at its new lower end binds in no run of any dataset (it did in
75%, 84%, 34% and 81% of the runs of INS-A, INS-I, INS-R and PEN with the control's range).
**Kept: low_variance becomes the control.** INSECTS moves from near zero to the level of
Mini-batch k-means in the real-data experiment (0.13 to 0.18).

## Round 2: formation of microclusters (control: low_variance)

The five m variants of commit f36ba01 and Chapter 3's chi-threshold, each on low_variance:
`m_const` (m = 3), `m_scalar` (constant m in [1, 5]), `m_param` (sigmoid with a in
[0.0007, 0.07] and b in [3, 1000], log), `m_reparam` (up in [1.05, 10], k95 in [5, 5000]),
`m_reparam_up3` (up = 3, k95 tuned), `chi_threshold` (q <= 1 + m sqrt(2/d)).

Prepared for round 3 (geometry and density, `geometry_round`) and round 4 (Chapter 3's remaining
candidates plus `no_noise` and `no_activation`, suggested by the diagnosis, `remaining_round`).
Cost check of round 3 on one run with 3 trials: the full covariance takes 91 s to tune and
116 s to evaluate on Gas (128 dimensions), about 50 minutes per Gas task at 100 trials, some
7 hours of Gas per covariance candidate on the notebook. Decision: round 3 runs the diagonal
engine and the densities first, and then a single covariance candidate, with the density that
the round keeps, instead of two.

Results (each against low_variance; Overall ARI medians, control > candidate):

- `m_const`: **-** (0 gains, 7 losses). Overall ARI INS-A 0.142 > 0.003, INS-I 0.137 > 0.007,
  INS-R 0.107 > 0.000, GAS 0.154 > 0.051, PEN 0.505 > 0.085, SHU unchanged; steady-state
  summaries of 1 to 6 structures. With m = 3 from the third sample on, microclusters absorb
  nearly everything and the model lumps the stream. The strict early m keeps microclusters
  apart; the hypothesis that young microclusters are too strict is not supported in this
  direction. Dropped.
- `m_scalar`: **-** (0 gains, 7 losses). Overall ARI INS-A 0.142 > 0.010, INS-I 0.137 > 0.008,
  INS-R 0.107 > 0.005, GAS 0.154 > 0.131, PEN 0.504 > 0.106; Shuttle unchanged. The search
  selects m near 1.2 to 2.1 (medians by dataset), and the summaries are several times smaller
  than the control's. A constant m, even close to the original sigmoid's value for young
  microclusters, loses the growth of m with S. Dropped.
- `m_param`: **-** (0 gains, 5 losses). Overall ARI INS-A 0.142 > 0.074, INS-I 0.137 > 0.037,
  INS-R 0.107 > 0.009, PEN 0.502 > 0.465; GAS and SHU unchanged. The search selects a and b
  close to the original constants (medians a 0.004 to 0.008, b 102 to 177, against 0.007 and
  100): the original sigmoid is near what the search prefers, and two more hyperparameters
  spread the same 100 trials, as Chapter 3 observed. Dropped.
- `m_reparam`: **-** (0 gains, 2 losses). Overall ARI INS-A 0.142 > 0.125, INS-I 0.130 > 0.065;
  higher but not significantly in GAS (0.154 > 0.170), PEN (0.504 > 0.528) and SHU (0.417 >
  0.440). The search selects a higher asymptote than 3 (up 3.9 to 5.7) reached more slowly
  (k95 640 to 1,700 samples; the original sigmoid reaches 95% of 3 near 520). Being analytically
  preferred keeps an indistinguishable candidate, not one with losses. Dropped.
- `m_reparam_up3`: **-** (0 gains, 2 losses). Overall ARI INS-A 0.142 > 0.118 and Steady-State
  ARI INS-I 0.164 > 0.083; the rest within noise. The search selects k95 of 630 to 1,500, a
  slower growth of m than the original sigmoid's. Dropped.

  So none of the five m variants helps: two make m constant and lose most (the growth of m with
  S matters), and the three that keep a growing m either re-find the original constants or
  prefer a slower growth, and lose on INSECTS.
- `chi_threshold`: **+ by the rule** (1 gain, 0 losses), **not kept: an override of the rule,
  flagged.** Its only significant change is a small gain in the Steady-State ARI of INS-I
  (0.164 > 0.175, delta 0.35). Its Overall ARI is lower in five datasets with deltas of -0.19
  to -0.30 (INS-A 0.142 > 0.124, p = 0.07 after Holm; INS-I, INS-R, GAS, PEN), none
  significant, and higher only in SHU (delta 0.15); its summaries are larger and it failed
  more tasks (5 against 3). The consistent direction says a slightly worse model with one
  favourable cell, so the conservative choice, which claims no improvement, is taken.

Round 2 keeps nothing: **low_variance remains the control.** On the real datasets the original
m function and the original threshold hold against every alternative tried.

## Round 3: geometry and density (control: low_variance)

First part: `diagonal` (one variance per feature), `volume_density` (S / sigma^2),
`log_density` (log of samples per isotropic volume), `diagonal_log_density`. Second part,
after them: a single full-covariance candidate, with the density that the first part keeps.

Results of round 3, first part (against low_variance):

- `diagonal`: **-** (0 gains, 5 losses). Overall ARI INS-A 0.142 > 0.093, INS-I 0.130 > 0.026,
  INS-R 0.107 > 0.007 (deltas -0.74 to -0.87), Steady-State ARI INS-I 0.163 > 0.000 and GAS
  0.076 > 0.033. Replay on INS-I: the search finds worse configurations already on the tuning
  segment (ARI 0.02 and 0.09 against the control's 0.17 and 0.18), one run settling on a large
  variance limit and a single cluster, the other labelling 23% to 30% of the samples as noise.
  Likely mechanism: the real classes vary in a few directions, so a maturing microcluster has
  tiny variances in most features, and the per-feature standardized distance turns small
  deviations in those features into outliers. Dropped. The full covariance shares the
  mechanism, more strongly, so before its eight hours a probe: `covariance` on runs 32-39 of
  INS-I, PEN and SHU (cheap datasets); the full run continues from it if the probe does not
  regress clearly.
- `covariance` probe (runs 32-39 of INS-I, PEN, SHU): INS-I collapses, ARI 0.000 in every
  shared run on both deciding metrics (0.142 and 0.164 for the control, delta -1.0,
  significant), with 5 microclusters at the end of the stream; PEN lower (deltas -0.78 and
  -0.89, not significant over 6 runs); SHU higher on the Overall ARI (0.426 > 0.548, delta
  0.41, not significant). The same failure as the diagonal engine, stronger. **Not run in full
  (flagged): neither covariance candidate continues.** The isotropic microcluster is retained
  on the real data, against the analytical expectation; the measured anisotropy is that of
  whole classes, and inside a microcluster the few directions of variance make the
  standardized distance over-strict in all the others.
- `volume_density` (S / sigma^2, Eduardo's order-independent density): **±** (1 gain, 1 loss).
  Gain: Steady-State ARI of SHU 0.365 > 0.473 (delta 0.40). Loss: Overall ARI of INS-A 0.142 >
  0.110 (delta -0.72). It also fails far more tasks by the summary limit (11 of 32 in PEN, 4 in
  INS-A, against 1 and 1), and its summary on INS-R grows fivefold (215 > 1,122); the
  mechanism of the INS-A loss was not examined beyond these counts. The gains do not outnumber
  the losses and the failures are a real cost. Dropped.
- `log_density` (log of samples per isotropic volume): **-** (0 gains, 6 losses). Overall ARI
  INS-A 0.142 > 0.092, INS-R 0.107 > 0.038, PEN 0.505 > 0.420, SHU 0.417 > 0.180 (deltas -0.57
  to -0.94); Steady-State ARI of PEN and SHU lower. Dropped.
- `diagonal_log_density`: **stopped before it ran (flagged).** Both of its parts regress
  strongly on their own, and the diagonal engine fails at forming microclusters whatever the
  density that decides the activation, so the pair cannot be kept; stopping saved some two
  hours.

Round 3 keeps nothing: **low_variance remains the control.** On the real data the isotropic
microcluster and the density of the last absorbed sample hold against both geometries and the
three densities.

## Round 4: remaining candidates (control: low_variance)

From the diagnosis first: `no_noise` (no noise label; the real classes have no noise) and
`no_activation` (every mature microcluster is active). Then Chapter 3's candidates outside its
final model: `mature_mean`, `link_factor` (l in [0.5, 4]), `recency`, `dim_penalty`
(2/d restored), `norm_typicality`, `local_weights`.

Results of round 4 (against low_variance):

- `no_noise`: **+** (1 gain, 0 losses). Overall ARI of INS-R 0.107 > 0.144 (delta 0.65);
  positive but not significant in INS-A (0.139 > 0.153, delta 0.49, p = 0.08), INS-I and GAS;
  slightly negative, not significant, in PEN (delta -0.20) and SHU (-0.04). More tasks fail by
  the summary limit (9 against 3) and the summary of INS-R is larger (215 > 709): the label
  acts on the prediction only, so the larger summaries come from the search, which without the
  noise distance tunes two hyperparameters instead of three. **Kept**: + by the rule, consistent
  with the absence of a noise class in these datasets, and one hyperparameter fewer. The other
  candidates of the round continue against low_variance, the control of the round; any of them
  that is kept is then tested together with no_noise.
- `no_activation`: **±** (2 gains, 6 losses). Gains in SHU only (Overall 0.409 > 0.449, Steady
  0.362 > 0.445); losses in the Overall ARI of INS-A, INS-I, INS-R, GAS, PEN (deltas -0.52 to
  -0.79) and the Steady-State ARI of INS-I. It fails 42 tasks by the summary limit and its
  summaries grow several times: only inactive microclusters decay, so with every mature one
  active almost nothing is pruned. The activation matters on the real data. Dropped.
- `mature_mean` (activation mean over mature microclusters only, from Eduardo's correction):
  **=** (no significant change on the deciding metrics). Analytically preferred, but its
  summary is significantly larger in INS-A (433 > 1,230 at the steady state) and it fails 11
  tasks against 3; the Overall ARI leans negative (INS-R delta -0.33, not significant). The
  condition of a summary not clearly larger is not met. Not kept.
- `link_factor` (intersection dist <= l (sigma_i + sigma_j), l in [0.5, 4]): **+ by the rule**
  (1 gain, 0 losses), **kept provisionally (flagged).** Gain: PEN Overall ARI 0.505 > 0.612
  (delta 0.51), and its fifth-batch ARI 0.346 > 0.530. But INSECTS leans negative: Overall
  deltas -0.35 (INS-A, p = 0.06), -0.16 (INS-I), -0.36 (INS-R), none significant. The search
  selects very different links (medians 0.85 in INS-R, 1.25 in PEN, 3.2 in SHU). Unlike
  `chi_threshold`, the gain is large; the lean on INSECTS is re-examined in the test of the
  combination with no_noise, and the candidate is dropped there if it turns into losses.
- `recency` (every microcluster not updated by a sample decays): **±** (2 gains, 3 losses).
  Gains in SHU (Overall 0.410 > 0.445, Steady 0.366 > 0.474); losses in the Overall ARI of
  INS-I (0.139 > 0.079) and INS-R (0.107 > 0.046) and the Steady-State ARI of INS-I. Gains do
  not outnumber losses. Dropped. Shuttle, dominated by one class, gains from the changes that
  hurt INSECTS (no_activation, recency).
- `dim_penalty` (the 2/d factor restored): **±** (2 gains, 4 losses). Gains in INS-I, large
  (Overall 0.137 > 0.148, delta 0.66; Steady 0.164 > 0.240, delta 0.95), and in the fifth-batch
  ARI of INS-A and INS-I (not deciding); losses in PEN (0.505 > 0.373) and SHU (0.417 > 0.192),
  on both metrics. Summaries about ten times smaller in every dataset (12 to 44 structures
  against 100 to 450) and no failure. Diagnosis from the engine: the factor cancels in the
  outlier test, the prediction and the noise label, where distances and variances are scaled
  alike, but not in the intersection condition, which compares unscaled distances between
  means with scaled deviations, so it acts as a link factor of about 4/d and microclusters
  almost never intersect. Good for INS-I, bad for the static datasets. Gains do not outnumber
  losses. Dropped; the much smaller summary is noted as the one clear cost advantage seen so far.
- `norm_typicality` (membership by normalized typicality): **-** (0 gains, 10 losses), every
  deciding cell, Overall ARI deltas -0.48 to -0.98 (INS-A 0.142 > 0.047, PEN 0.502 > 0.337).
  The nearest-microcluster membership of Chapter 3 holds on the real data. Dropped.
- `local_weights` (membership weighted by the density at the sample): **-** (0 gains, 9 losses),
  like norm_typicality (Overall INS-A 0.142 > 0.049, PEN 0.505 > 0.360). Dropped.

Round 4 keeps `no_noise`, and `link_factor` provisionally; the control of round 5 is no_noise.

## Round 5: on no_noise (registered while round 4 finishes)

- `no_noise_link`: the change of link_factor on no_noise, the test of the combination that
  decides whether link_factor, kept provisionally, stays.
- `no_noise_link_wide`: the intersection factor searched on a log scale in [0.05, 4]. Reason:
  dim_penalty showed that a link of about 4/d (0.12 in INSECTS) helps INS-I, below the range of
  link_factor, while link_factor's larger links helped PEN; a wider range lets the search
  choose per dataset.
