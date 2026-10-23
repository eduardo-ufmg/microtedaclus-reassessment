# Stage 1 Outline: Structure and Cross-References

This outline is for your approval before any rewriting. It assumes the conventions in `STYLE.md`, the comparison of the model in focus with each benchmark model, and the decisions already taken: Chapter 5 becomes part of Chapter 4, and the appendix keeps only the tables Chapter 4 cites. Approved on 2026-10-22, with every recommendation accepted and the real-data chapter titled "Real Data".

## Quality metrics (decided 2026-10-23)

Only the Overall and Steady-State ARI are reported. The ARI at the fifth batch and the maximum ARI drop leave every table, figure and sentence, with two exceptions that keep their evidence through trajectories:
- the slow start of the final MicroTEDAClus+ on the real streams with drift, one of its remaining limits, gets a figure of its ARI along those streams in Chapter 7, drawn from the per-batch results we already have;
- DenStream's late start in Chapter 5 rests on `fig:mod_stream`, which already shows it.

Chapter 4's three tables of non-deciding metrics disappear.

## Chapter list after the revision

| New | Old | Title | Change |
|---|---|---|---|
| 1 | 1 | Introduction | Roadmap updated |
| 2 | 2 | Background and Related Work | Forward links; headings flattened |
| 3 | 3 | Methodology Reassessment | Reordered; a single model set; prose about MicroTEDAClus only |
| 4 | 4 + 5 | Model Improvement | Ends with one comparison in the main experiment |
| 5 | 6 | Experiment Modernization | Comparison with each benchmark model; forward links |
| 6 | 7 | Real Data (was "Real Data Datasets") | Comparison with each benchmark model; forward link to Chapter 7 |
| 7 | 8 | Model Improvement on Real Data | Comparison with each benchmark model |
| 8 | 9 | Discussion | Rewritten on the new comparisons |
| 9 | 10 | Conclusions | Rewritten on the new comparisons |
| A | A | Complementary Tables of the Model Improvement Experiments | Three tables, all cited |

Chapter numbers shift from the old Chapter 6 on, so prose that names chapters must use `\cref`. A search for hard-coded "Chapter N" is part of Stage 5.

## Chapter 2 — Background and Related Work

- **Structure:** keep the four sections. Turn the subsubsections into `\paragraph` headings.
- **Forward links**, one sentence each, from each issue to where it is tested or resolved:
  - The order-dependent density goes to Chapter 4's density corrections (*peak-density*, *volume-density*) and to their second test in Chapter 7.
  - The two-sample condition goes to Section 3.1, which shows that the reference implementation already uses the variance alone.
  - The dimensionality penalty goes to the collapse of MicroTEDAClus in high dimension, shown in Section 3.8, and to *no-penalty* in Chapter 4.
  - Galindo's decay goes to Section 3.1, since the reference implementation contains it, and to *recency* in Chapter 4.
  - Cardoso's factor $\phi$ goes to *link-factor* in Chapter 4 and to the kept link factor in Chapter 7.
- **Notation:** apply Section 9 of the style sheet to the equations.

## Chapter 3 — Methodology Reassessment

The critique sections keep their order. The methodology we use is gathered just before the results it produces, and the results keep one model set.

| New section | Comes from | Change |
|---|---|---|
| 3.1 Model Implementation | 3.1 | Listings cut to the lines that carry each argument; listings that make one point are merged |
| 3.2 Benchmark Models | 3.2, without its tuning and runs subsections | The original work's interface layers, our implementations, the TEDA-based models, and the final set of six. The six settings tables (streamMOA, River) become one table |
| 3.3 Datasets | 3.3 | The preprocessing experiment keeps its evidence in two panel figures, one of scatter plots and one of metric curves, instead of ten. Concept drift and scalability datasets as now |
| 3.4 Evaluation Protocol | 3.5 | The original work's protocols, then the prequential protocol we use |
| 3.5 Metrics | 3.4 | Only the Overall and Steady-State ARI, under the names used later. The hypothetical-curves figure stays to motivate the Steady-State ARI and the trajectory plots; the complementary-scores table goes. Then execution time, memory footprint and microcluster count |
| 3.6 Hyperparameter Tuning and Runs | 3.2.4, 3.2.5 | Moved here. Forward links: Chapter 4 probes the limits of the search, and Chapter 5 replaces its ranges and lengthens the tuning segment |
| 3.7 Statistical Comparison | 3.6 | The comparison with each benchmark model, the variant-against-control comparison used in Chapters 4 and 7, and the single "reading the tables" paragraph |
| 3.8 MicroTEDAClus Under the Reassessed Methodology | 3.7 | One set: MicroTEDAClus against the six benchmark models. Clustering Quality, Cost, Microcluster Count, Scalability. Prose only about MicroTEDAClus. Forward links from time and memory to the isolated measurements of Chapters 4 and 5 |
| 3.9 Hyperparameters of MicroTEDAClus | 3.8 | Only MicroTEDAClus's dispersion, which bears on the original work's robustness claim. The benchmark models' dispersion analysis is dropped |
| 3.10 Summary | new | Short |

**Removed:**
- the original-work and TEDA-based model sets: 20 tables, 32 scalability plots and their prose;
- the benchmark models' parameter-variability tables and prose;
- all prose that ranks the benchmark models.

## Chapter 4 — Model Improvement, including the former Chapter 5

| Section | Change |
|---|---|
| 4.1 Experimental Design | As now |
| 4.2 Implementation Optimization | Cite the speedup table: the text discusses it without citing it, and the figure shows something else, the speedup against the microcluster count |
| 4.3 Corrections | As now; cost tables go per the appendix decision |
| 4.4 Improvements | Refinements, covariance and densities revisited stay. Covariance and densities are compressed in Stage 3. Forward links from every candidate that Chapter 7 tests again: *m-constant*, *m-reparam*, *chi-threshold*, *recency*, *no-activation*, *link-factor*, *covariance*, the two densities, *norm-typicality*, *local-weights* |
| 4.5 Validation | As now; it cites one of the three tables kept in the appendix |
| 4.6 Limits of the Hyperparameter Selection | Forward link to Chapter 5's ranges by rule and longer tuning segment, which answer it |
| 4.7 MicroTEDAClus+ | Adds the naming sentence: Chapter 7 revises the model, and from there on the name means the revision |
| 4.8 MicroTEDAClus+ in the Main Experiment | One comparison with MicroTEDAClus+ in focus against MicroTEDAClus and the six benchmark models. It replaces both the old comparison with MicroTEDAClus (10 tables) and the former Chapter 5 (10 tables). Under the new design each pairwise result has the same form, so the original model is simply one more model compared. Subsections: Clustering Quality, Cost, Microcluster Count, Scalability, Hyperparameters of MicroTEDAClus+ (from the former 5.5), Summary |

The former 4.8's discussion of the search budget against MicroTEDAClus stays, as a note within 4.8.

## Chapter 5 — Experiment Modernization (was 6)

- **Structure:** as now.
- **Results:** recast as the comparison with each benchmark model. Keep the protocol comparison, which is already a variant-against-control comparison.
- **Benchmark material:**
  - Keep what bears on fairness: binding search ranges, and DenStream's late start.
  - Drop prose that ranks the benchmark models, such as BICO's learning time, MiniBatchKMeans's peak, and how much memory BIRCH holds.
- **Forward link:** the binding ranges go to the widened rules of Chapter 6.

## Chapter 6 — Real Data (was 7)

- **Title:** "Real Data".
- **Structure:** as now. Comparison of MicroTEDAClus+ of Chapter 4 with each benchmark model.
- **Benchmark material:** keep BIRCH's failures and the rule for leaving models out, since both bear on the comparison.
- **Forward link:** the weak results go to Chapter 7's diagnosis.

## Chapter 7 — Model Improvement on Real Data (was 8)

- **Structure:** as now.
- **Comparison with the Benchmark Models:** recast as the comparison with each benchmark model.
- **Naming:** the final version is MicroTEDAClus+, and the earlier one is MicroTEDAClus+ of Chapter 4.

## Chapters 8 and 9 — Discussion and Conclusions

- **Structure:** as now.
- **Rewrite** every comparative claim on the new results, as better than, indistinguishable from, or worse than each benchmark model. Re-check the Limits of the Evidence.

## Introduction, Abstract, Resumo

- Revised last.
- The roadmap follows the new chapter list.
- The results sentences use the new comparisons.
- Keywords are added to the Abstract and the Resumo.

## Appendix A

- Keeps the cost tables of Implementation Optimization, Covariance and Validation, all cited from Chapter 4.
- Its opening sentence is updated.

## Expected effect

These are estimates for Stage 2 to confirm:
- Chapter 3 falls from about 100 floats to about 25, and from 64 pages to roughly 30.
- Chapter 4 plus the former Chapter 5 loses about 30 tables.
- The List of Tables falls from 137 entries, 50 of them duplicates, to about 60, with no duplicates.
- The thesis would fall from 231 pages to roughly 150–170.

## Order of work after approval

1. **Stage 2**, artifacts:
   - mock-ups of the comparison tables for your choice;
   - then generator changes, chapter by chapter: Chapter 3 (old generator), Chapter 4 including the former Chapter 5 (old generator), then Chapters 5–7 (`benchmark` package);
   - I commit code and artifacts; the text changes only in its `\input` lines.
2. **Stage 3**, prose: one chapter per review round, in the order 3, 2, 4, 5, 6, 7, 8, 9, Introduction, Abstract, Resumo.
3. **Stage 4**, front and back matter, alongside Stage 3.
4. **Stage 5**, scripted checks and a final read.
