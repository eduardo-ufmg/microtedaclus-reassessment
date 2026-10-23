# Style Sheet for the Thesis Revision

Applies to every chapter, the front and back matter, and the generated tables and figures.
Where this sheet and an older rule disagree, this sheet wins. Every item here was accepted on 2026-10-22.

## 1. Precedence

- Standard English academic writing is the default. ABNT applies where the course requires it: the cover, the title page, and the Resumo in Portuguese, in the third person singular (NBR 6028).
- Anything standard in English academic writing is allowed, even where it breaks an earlier rule of ours.

## 2. Voice and tense

- Use "we" for our actions, choices and reasoning. Results keep their natural subjects, for example "MicroTEDAClus+ is faster than…" or "Table 4.3 shows…".
- Present tense for procedures and for what the evidence shows. Past tense for what happened during the improvement efforts, for example "we did not run…".
- The Abstract uses "we"; the Resumo uses the third person singular.

## 3. Spelling and wording

- American spelling throughout: behavior, favor, criticized, labeled, modeling, color, analyze.
- Fix the typos found in Chapters 2 and 3, for example colapse, suposed, stricly, Scalling, ploted, midle, indistiguishable, identicaly, hypotesis, exibit, enviroments, batchs, accross, "does not depends".
- Prefer the short, plain word: "a single algorithm", not "one same algorithm"; "to", not "in order to".

## 4. Parentheses, values and pointers

- Parentheses are allowed where English papers use them: pointers such as "(Table 4.3)" or "(Section 3.6)", abbreviations at first use, and short clarifications. Long asides stay out of parentheses.
- Values in prose are allowed when they carry the argument, for example "about ten times faster" or "an Overall ARI above 0.9". Use them sparingly; the tables remain the record. I read this as covered by "anything that is standard".
- An observation or explanation still needs a table, figure or citation beside it.

## 5. Cross-references

- Use `\cref`, and `\Cref` at the start of a sentence. Load cleveref with `[capitalise,noabbrev]`, so it prints "Table 4.3", "Figure 3.2", "Section 3.6", "Chapter 4" and "Equation (2.9)" instead of today's "table 4.3" and "eq. (2.9)".
- Forward references are allowed. Use one when it replaces a re-explanation, or when it links an issue to the place where it is tested or resolved.
- Point to a chapter or section instead of repeating its content.

## 6. Citations

- When a work is the subject of a sentence, name the author:
  - one author: "Galindo [5] updates…";
  - two authors: "Hyde and Angelov [n]…";
  - three or more: "Maia et al. [3]…".
- Otherwise the citation closes the clause it supports, for example "…a standard practice [12, 13]."

## 7. Names and typography

- Model and dataset names are plain text. The macros `\modelname` and `\datasetname` stop producing bold, which is a one-line change each.
- Variants of the improvement efforts, such as *low-variance* or *no-noise + link-factor*, are set in italics through a new `\variant` macro, so compound names stay readable in prose.
- Code identifiers, hyperparameter keys and file names use typewriter (`\codename`).
- Bold is reserved for the statistical marks in tables.
- Names:
  - MicroTEDAClus is the original model.
  - MicroTEDAClus+ is the model defined in Chapter 4. From the chapter on model improvement with real data on, the name means the final version, and the earlier one is "MicroTEDAClus+ of Chapter 4" ("MicroTEDAClus+ (Ch. 4)" in tables). Chapter 4 says this once, where it defines the model.
  - "Benchmark models" names the models we compare against; never "opponents" or "competitors".

## 8. Terminology

| Use | Not |
|---|---|
| hyperparameter, for tuned settings | parameter |
| Overall ARI; Steady-State ARI, meaning the ARI at the penultimate batch. These are the only quality metrics reported; the ARI at the fifth batch and the maximum ARI drop were dropped on 2026-10-23 | penultimate batch's ARI |
| adjusted Rand index, spelled out once, then ARI | adjusted rand index |
| learning stage and prediction stage | fitting stage, fitting |
| execution time | complete runtime, runtime |
| memory footprint, maximum and steady-state, for the measurement of Chapters 3 and 4; peak memory and memory held at the end of the stream, for the measurement from Chapter 5 on | memory usage, memory consumption |
| microcluster count | number of microclusters, MicroCluster count |
| microcluster, macrocluster | micro-cluster |
| dataset; stream; scalability streams | data set |
| batch, only for the 100-sample processing batches | batch, for any other grouping |
| run; development runs; validation runs | — |
| Euclidean | euclidean |
| the Chameleon datasets by their macros; ST-D1 and ST-D2 only when quoting the original work | — |

## 9. Notation

| Symbol | Meaning |
|---|---|
| $k$ | time index |
| $d$ | dimension |
| superscript $i$ | microcluster index |
| $S^i_k$ | sample count |
| $\mu^i_k$ | mean |
| $\sigma^i_k$, $(\sigma^i_k)^2$ | standard deviation, variance |
| $\xi$, $\zeta$ | eccentricity, normalized eccentricity |
| $\tau$, $t$ | typicality, normalized typicality |
| $D$ | density |
| $m(\cdot)$ | sensitivity function |
| $r_0$ | absolute variance limit |
| $r$ | relative variance limit |
| $\rho = 1/\text{decay}$ | decay rate |
| $\ell$ | link factor |
| $n_\sigma$ | noise distance |
| $q$ | standardized squared distance |
| $\phi$ | Cardoso's intersection factor |

- One form per quantity. For example, fix Chapter 2's mix of $\sigma^2_k$ with $(\sigma^i_k)^2$, and the stray $u^i_k$ in the dimensionality-penalty equations.
- Displayed equations stay numbered.

## 10. Statistical comparisons

- Two kinds of comparison, both defined once in Chapter 3:
  - **Model in focus against each benchmark model.** Paired Wilcoxon signed-rank test, Holm correction across the benchmark models, and a difference counts only when Cliff's $|d| \ge 0.147$. This follows Demšar's comparison with a control.
  - **Variant against its control**, as in Chapters 4 and 8.
- The discretization of the metrics before testing stays as it is.
- Marks always describe the model under test relative to its reference: bold means significantly better, red means significantly worse. Stage 2 will show mock-ups of the layout for the comparison tables before any generator changes.
- Chapter 3 holds one "reading the tables" paragraph. Captions don't repeat it.

## 11. Tables

- Models as rows and datasets as columns. Related metrics go in panels of one table, with medians only.
- Hyperparameters as `\multirow` parents of their model, with "median [QCD]" cells.
- Scalability streams in separate tables or figures from the six datasets. There are no complete quality tables for scalability streams.
- Dataset abbreviations appear only in table headers, defined once in a small table, one for the synthetic datasets and one for the real ones.
- Three decimals for ARI. Time in ms and memory in KiB, with the unit in the header.
- Generated tables are set in `\small` at their natural size, reduced only when wider than the text block.

## 12. Figures

- The model in focus is drawn in red, with its interquartile band. Each benchmark model keeps one color in every chapter and is drawn as a thin median line without a band (`pfc_shared.figures`).
- Axis labels carry their units. The dimension axis of the scalability streams is logarithmic in base 2.
- Related plots are grouped as panels of one figure.

## 13. Captions

- At most two sentences: what is shown, and over which runs.
- A third, short sentence is allowed only for a note specific to that table, such as what a dash means.
- Long captions get a short entry for the lists through `\caption[short]{long}`.
- No caption repeats the conventions of Section 10.

## 14. Headings and structure

- Title case. At most `\section` and `\subsection`; use `\paragraph` below that.
- Recurring headings use the same words in every chapter: "Clustering Quality", "Cost" (time and memory), "Microcluster Count", and "Summary" closing each experimental chapter.
- Each chapter opens with a short paragraph saying what it does and how it connects to the others.

## 15. Numbers and units

- Numbers below ten in words, except with units or in a series that includes larger ones.
- SI units with a thin space, for example 10\,ms.
- Ranges as $[a, b]$; run ranges as "runs 0 to 31".

## 16. Front and back matter

- Order follows the course template: cover, title page, Resumo, Abstract, Acknowledgements, Contents, List of Figures, List of Tables.
- Keywords: "Palavras-chave:" after the Resumo and "Keywords:" after the Abstract.
- The lists are titled "List of Figures" and "List of Tables", and the bibliography "References".
- The same title and name casing on the cover and the title page, and a fixed date: month and year of the defense.
- Figure captions below the figure and table captions above the table, with no "Fonte:" lines.
