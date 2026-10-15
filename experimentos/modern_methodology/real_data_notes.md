# Real-data experiment: decisions and hand-off

Written while Eduardo was away (2026-10-15), on the branch `real-data`, for his review.
Every decision below is flagged **[D#]** so that it can be discussed. Nothing was pushed.

## Status

Ready for the remote run; not started. The code is committed on `real-data`, the hand-off
archive was tested end to end in a scratch folder (see "Tests"), and the commands are at the
end. The chapter itself is not written yet. The three decisions that most deserve a look are
**[D8]** (tuning segment capped at 10,000 samples), **[D10]** (summary limit, under which a
model can fail a task) and **[D12]** (KDD Cup 99 shuffled).

## Decisions

**[D1] Branch.** The work is on a new branch, `real-data`, from `main` at 49b079c, so that
it can be reviewed or dropped as a whole. Chapter 5's code paths and results are unchanged:
the synthetic datasets, their splits and the rules of Chapter 5 give the same values as
before (checked against the stored search ranges and splits).

**[D2] Data files.** The eight datasets were downloaded once, with his approval, into
`data/real/` (git-ignored), and their SHA-256 checksums are recorded in
`benchmark/real_data.py`, which checks them before reading. KDD Cup 99 and Covertype come
from the mirror that scikit-learn uses and match its published checksums. Shuttle's training
file comes compressed by Unix `compress`, which Python does not read; it is kept decompressed
(`gzip -dc shuttle.trn.Z > shuttle.trn`) and its checksum is that of the decompressed file.
The files travel in the hand-off archive, so the remote run needs no internet.

**[D3] Features and labels.** As the documentation of each dataset describes them:
INSECTS 33 features, Gas 128, KDD 99 the 34 continuous ones (symbolic and binary left out),
Covertype the 10 quantitative ones, Pendigits 16, Shuttle the 8 after the first, which is
time. Labels are the given classes as integers: the 23 attack types of KDD 99, the six gases,
the six insect classes. No dataset has a missing, infinite or non-numeric value; every Gas
line has its 128 entries; the readers stop with an error otherwise.

**[D4] Order.** INSECTS and Covertype keep their order. Gas keeps the order of its ten
batches and is shuffled within each batch in every run. Pendigits, Shuttle (whose
documentation says its order was already randomised) and KDD 99 ([D12]) are shuffled in
every run.

**[D5] Standardization.** By the mean and standard deviation of the tuning segment, applied
to the whole stream, after dropping the features constant in the tuning segment. Only KDD 99
loses features: `num_outbound_cmds`, zero throughout the dataset, and, depending on the run,
two or three of its rarest counters when the tuning sample happens to hold none of their
nonzero values (31 of 34 kept in run 0). Large standardized values after the tuning segment
exist (drift) but are rare: in KDD 99 in its recorded order, 0.7% of the evaluation samples
had some value beyond 20 standard deviations.

**[D6] Widened rules**, as approved, in `benchmark/hyperparameters.py`. Minimum points are
real numbers from 1; DenStream divides by zero at exactly 1, which the search practically
never samples, and a trial that raises an error is scored as failed, as before.

**[D7] Median distance between two samples** from 1,000,000 seeded pairs in the widened
rules (all pairs of a 10,000-sample segment would take 400 MB per computation).

**[D8] Tuning segment capped at 10,000 samples (main new decision).** The tuning segment is
the first fifth of the stream, up to 10,000 samples, the longest tuning segment of the
synthetic experiment, whose splits are therefore unchanged. Evaluation still covers the whole
rest of every stream. Affected: KDD 99 (would be 98,804), Covertype (116,202), the three
INSECTS streams (10,569 to 15,997) and Shuttle (11,600); Gas and Pendigits are shorter.
Why: at the fine end of the widened ranges, five of the six models keep about one structure
per sample, and their cost grows faster than the segment (probe below). On KDD 99, one trial
of 8,000 samples already took 17.7 s for MicroTEDAClus+, 20 s for BIRCH and 47 s for
DBSTREAM, growing about quadratically; a trial over the full 98,804-sample segment would take
from tens of minutes to hours and gigabytes, and a search has 100 trials. With the cap, the
costliest corner trials take about one minute, the regime of Chapter 5. Alternative
considered and rejected: a per-trial time limit, which would make the selection depend on the
machine, as Chapter 5 avoided. To disclose in the chapter, like the 32 runs: the cap is a
matter of cost.

**[D9] Guards for the unattended run.** Each worker gets a memory limit (`--memory-limit`, in
gigabytes); a task that exceeds it fails with a MemoryError and is recorded as failed,
instead of exhausting the machine. The stall limit of the run is three hours, far above the
longest task expected (about ten minutes): a run that finishes no task for three hours ends
with an error.

**[D10] Summary limit (main new decision).** Under the widened rules, no model may keep a
larger summary than the largest that the summary-size rule allows: a quarter of the tuning
segment (2,500 structures for a 10,000-sample segment; 695 for Gas, 549 for Pendigits). A
trial whose summary grows beyond it fails, as a trial that raises an error already did; a
selected model whose summary grows beyond it during the evaluation fails its task, which is
recorded as failed and not repeated. Why: the first smoke run showed that the evaluation is
the larger risk. BIRCH never forgets, and scikit-learn re-clusters all of its subclusters
after every batch, at a cost that grows with their square (1.1 s at 5,000 subclusters, 4.8 s
and 1 GB at 10,000, measured); over the 484,021-sample evaluation segment of KDD 99, a BIRCH
that keeps 10,000 subclusters would take about six hours for one task, and BIRCH failed for
memory on INSECTS in the first smoke run. The limit depends on the summary only, not on the
machine, and bounds both time and memory: at 2,500 structures BIRCH re-clusters in about
0.6 s, and the fine-grained trials stop early. Consequences to disclose: configurations that
keep more than a quarter of the tuning segment are excluded from the search, and a model can
fail a task. In the smoke runs (3 trials per search), 3 of 48 tasks failed by the limit:
BIRCH on INSECTS reoccurring and on KDD 99 (in both orders), DBSTREAM on Gas. Expect BIRCH, which
never forgets, to fail often on the long and drifting streams; that is a finding to report,
not an error. Alternative: a limit of the whole tuning segment (10,000), with BIRCH tasks of
up to six hours each. Applies to the real datasets only; Chapter 5 is unaffected.

**[D11] Tables.** `benchmark/real_artifacts.py` writes, under `monografia/RealData/artifacts/`,
the quality table (four ARI panels), the summary-size table, the table of the ends of the
widened ranges, a table of the number of failed runs of every model in every dataset, and a
figure of the batch ARI along the abrupt INSECTS stream with its change points. A model that
failed in more than half of the runs of a dataset is left out of that dataset (its cells show
a dash), and the other models are compared over the runs that all of them finished, so a
failed task removes its run from its dataset only. Tested on the smoke results.

**[D12] KDD Cup 99 shuffled (main new decision).** In its recorded order, 96.5% of the
100-sample batches of KDD 99's evaluation segment hold a single class (the largest class
holds 99.2% of a batch on average): long runs of one attack. A batch of one class scores an
ARI of 1 if the model puts it in one cluster and 0 if it splits it, so the batch ARI measured
little but lumping; in the smoke run four models scored an Overall ARI of exactly 1.000 and
Mini-batch k-means 0.000. This is the flaw that Chapter 2 found in the original experiments,
whose ordering made most batches hold one cluster. Shuffled in every run, no batch holds a
single class, the largest class holds 57% of a batch on average, and the tuning segment holds
12 of the 23 classes instead of 6. What is lost: the temporal structure of KDD 99, which the
stream-clustering literature uses as evolution. Alternatives: keep the recorded order and
explain why its ARI is uninformative, or drop KDD 99. The other streams were checked the same
way: at most 11% single-class batches (INSECTS reoccurring, whose class proportions change by
design) and none in most.

## Probe of the costliest configurations

One tuning trial (a prequential pass over the tuning segment) at the costly corner of each
model's widened space: the finest distance or variance, the longest memory, the lowest
minimum points, the most clusters or the largest summary. Each trial ran alone, in its own
process, with a memory cap and a time limit, before the summary limit existed (with it, these
trials now stop as soon as the summary passes the limit). Seconds per trial over the capped
segment:

| Dataset (tuning samples) | MicroTEDAClus+ | DenStream, longest buffer | DBSTREAM | BIRCH | BICO |
|---|---|---|---|---|---|
| KDD 99, recorded order (10,000) | 17 | 44 | 67 | 43 | 0.5 |
| Covertype (10,000) | 5 | 26 | 41 | 42 | 1.4 |
| INSECTS reoccurring (10,000) | 10 | 63 | 79 | 53 | 0.4 |
| Shuttle (10,000) | 13 | 20 | 35 | 31 | 0.5 |
| Gas (2,782) | 3 | 0 | 18 | 2 | 0.1 |

Mini-batch k-means stays under a second. At these corners the summaries hold about one
structure per sample; the highest peak of memory was 1.2 GB (BIRCH). Before the cap, on the
uncapped KDD 99 segment, the same corners grew about quadratically: 8,000 samples already
took 17.7 s (MicroTEDAClus+), 20 s (BIRCH) and 47 s (DBSTREAM), on the way to 98,804.

## Tests

- The hand-off archive was unpacked into a scratch folder and run through
  `run_real_experiment.sh` itself, with `--runs 0 0 --trials 3 --workers 3 --memory-limit 2.5`:
  48 tasks in 2.4 minutes, "The experiment is complete.", no process left. Run again with
  `--runs 0 1` on three datasets, it skipped the 48 finished tasks and ran the 18 new ones.
- `python -m benchmark.real_artifacts` produced every table and the figure from those results.
- Stopping the runner alone with SIGTERM, mid-run, ended it and its workers, kept the 22
  finished tasks, and the same command resumed and completed the other 26 (checked on the
  final archive).
- In the figure of the smoke run, BIRCH scores 1 on stretches of the abrupt INSECTS stream
  where the batches hold one class (6.5% of its batches) and BIRCH puts each in one cluster:
  the effect of [D12], small in this stream; worth a sentence in the chapter if it persists.

## Expected duration

About 2 hours on 16 workers: the smoke run's tuning times scaled from 3 to 100 trials, plus
its evaluation times, give about 60 CPU-minutes per run on the notebook, times 32 runs, over 16
workers; the longest task about 9 minutes. The estimate is rough (the search concentrates on
other configurations than three random trials, and the machines differ), so allow up to about
10 hours; the stall limit ends a run that stops advancing for three hours.

## Observation, not investigated

Chapter 5's Random RBF stream (synthetic) holds 40,013 of its 50,000 samples in one cluster
(the rest are noise and five small clusters), so its batches mostly hold one cluster and
noise. Its generator was validated against R's in Chapter 5; it is noted here only because
the check of [D12] showed it.

## Hand-off

Archive: `real_experiment.tar.gz` (46 MB), the committed code of `experimentos/modern_methodology`
and `experimentos/shared` (`git archive`) plus the nine data files. On the remote machine,
from the project folder that holds `experimentos/` and the `.venv` of the previous run:

```bash
tar -xzf real_experiment.tar.gz
cd experimentos/modern_methodology
nohup ./run_real_experiment.sh --workers 16 --memory-limit 5 > run_real.log 2>&1 &
```

16 workers with 5 GB each use at most 80 of the machine's 94 GB. No new package is needed.

- Progress: `tail -n 3 run_real.log` (a line every 50 of the 1,536 tasks).
- Finished: the log ends with `The experiment is complete.`
- Stalled: the log ends with `ERROR: no task finished in 10800 seconds`.
- Failed tasks are listed in the log as `DATASET MODEL RUN: ERROR: ...`; failures by
  `SummaryLimitExceeded` are expected ([D10]) and the run continues.
- Stop: `pkill -TERM -f "benchmark.runner results_real"`; then `pgrep -af benchmark.runner`
  should print nothing. Resume: the same `nohup` command; finished tasks are skipped.
- Bring back: `tar -czf real_results.tar.gz results_real run_real.log`, to be unpacked in
  `experimentos/modern_methodology/` (where `results_real` does not exist yet).

## For the chapter, later

Dataset table with references (Souza et al. 2020; Vergara et al. 2012; the KDD Cup 1999 data
and Tavallaee et al. 2009; Blackard and Dean 1999; Alpaydin and Alimoglu 1996; Statlog
Shuttle; the UCI repository); the 32 runs, the tuning cap and the summary limit as matters of
cost; the standardization by the tuning segment as the non-leaking form of the scaling that
Chapter 2 discusses; KDD 99's shuffle; the ends of the widened ranges; the failures.
