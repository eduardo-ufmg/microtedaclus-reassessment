r"""
Normality comparison table (LaTeX).

Reads the per-run CSVs produced by normality_test.py (results/<family>_<n>.csv,
one file per (family, n_total) dataset) and builds a single hierarchical
LaTeX table (results/normality_table.tex):

    run quantity (20, 100)
      -- dataset samples (4000, 8000)
           --- dataset (SCL2, Benchmark1, Benchmark2)

12 body rows total (2 x 2 x 3), grouped via \multirow in the first two
identifying columns. To the right, one column per metric (ari,
total_time_ms, memory_bytes, microcluster_count), holding ONLY the
Holm-adjusted Shapiro-Wilk p-value (p_holm) -- see chat discussion: W is a
sample-size-dependent goodness-of-fit measure with no fixed critical value
(not comparable across the n=20 / n=100 rows), and raw p is invalid once
48 simultaneous tests are being reported. p_holm is the only quantity that
is both scale-calibrated and valid under repeated testing, so it is the
only value shown.

ASSUMPTIONS (flagged per your preference, please confirm/correct):
  - p_holm is bolded whenever p_holm < 0.05 (rejects normality).
  - n_ok / n_attempted (present in the earlier CSV version) are dropped
    from the table body to match the requested column layout.
    Incomplete-data situations (a dataset with fewer than `sample_count`
    stored/OK runs) still print a console warning, and the corresponding
    cell is rendered as "--" rather than computed on a truncated sample.
  - \footnotesize is applied to the tabular; no \resizebox or landscape is
    added, since the requested skeleton didn't include one -- add one
    yourself if it still overflows \textwidth (now only 7 columns wide,
    so likely unnecessary).
  - The output assumes \usepackage{multirow} is available in the
    including document (required for the row-spanning cells).
  - p_holm is formatted in LaTeX scientific notation ($a \times 10^{b}$)
    when < 1e-3, otherwise as a plain 3-sig-fig decimal. Bold uses
    \boldsymbol inside math mode (scientific case) or \textbf in text mode
    (decimal case) -- \textbf alone does not bold math-mode content, and
    \boldsymbol (vs. \mathbf) also bolds the \times symbol, not just the
    digits. \boldsymbol requires the amsmath package.
  - W and raw p are still computed internally (W feeds nothing further;
    raw p is required as Holm's input) but are no longer written to the
    table.

Output: results/normality_table.tex (LaTeX only).
"""

import os

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

RESULTS_DIR = "./results"
OUTPUT_PATH = os.path.join(RESULTS_DIR, "normality_table.tex")

FAMILIES = ["SCL2", "STR-B1", "STR-B2"]
DATASET_SIZES = [4000, 8000]
SAMPLE_COUNTS = [20, 100]
METRICS = ["ari", "total_time_ms", "memory_bytes", "microcluster_count"]
METRIC_LABELS = {
    "ari": "ARI",
    "total_time_ms": "Total time (ms)",
    "memory_bytes": "Memory (bytes)",
    "microcluster_count": "Microcluster count",
}

ALPHA = 0.05

os.makedirs(RESULTS_DIR, exist_ok=True)


def csv_path(family, n_total):
    return os.path.join(RESULTS_DIR, f"{family}_{n_total}.csv")


# --------------------------------------------------------------------------
# Data collection (Shapiro W / p per (run_quantity, n_total, family, metric))
# --------------------------------------------------------------------------
def collect_records():
    records = []
    warnings = []

    for s in SAMPLE_COUNTS:
        for n_total in DATASET_SIZES:
            for family in FAMILIES:
                dataset_label = f"{family}"
                path = csv_path(family, n_total)

                row = {
                    "run_quantity": s,
                    "n_total": n_total,
                    "dataset": dataset_label,
                }

                if not os.path.exists(path):
                    warnings.append(
                        f"[{family}_{n_total}] results file not found — skipped."
                    )
                    for metric in METRICS:
                        row[str((metric, "W"))] = np.nan
                        row[str((metric, "p"))] = np.nan
                    records.append(row)
                    continue

                df = pd.read_csv(path, index_col=0)
                df_ok = df[df["status"] == "OK"] if "status" in df.columns else df

                attempted = df.loc[df.index <= s]
                subset_ok = df_ok.loc[df_ok.index <= s]

                if len(attempted) < s:
                    warnings.append(
                        f"[{family}_{n_total}, run_quantity={s}] only {len(attempted)}/{s} "
                        f"runs stored so far — cell left blank."
                    )
                    for metric in METRICS:
                        row[str((metric, "W"))] = np.nan
                        row[str((metric, "p"))] = np.nan
                else:
                    if len(subset_ok) < s:
                        warnings.append(
                            f"[{family}_{n_total}, run_quantity={s}] {s - len(subset_ok)} of {s} "
                            f"attempted runs had status != 'OK' — Shapiro computed on the "
                            f"remaining {len(subset_ok)} OK runs."
                        )
                    for metric in METRICS:
                        data = subset_ok[metric].dropna().values
                        if len(data) >= 3:
                            w, p = stats.shapiro(data)
                        else:
                            w, p = np.nan, np.nan
                        row[str((metric, "W"))] = w
                        row[str((metric, "p"))] = p

                records.append(row)

    return records, warnings


def add_holm_correction(records):
    p_positions = []
    raw_pvals = []
    for i, row in enumerate(records):
        for metric in METRICS:
            p_val = row[str((metric, "p"))]
            if not np.isnan(p_val):
                p_positions.append((i, metric))
                raw_pvals.append(p_val)

    if raw_pvals:
        _, p_holm_vals, _, _ = multipletests(raw_pvals, method="holm")

        if p_holm_vals is None or len(p_holm_vals) != len(raw_pvals):
            raise ValueError("Holm correction failed: output length mismatch.")

        for (i, metric), p_holm in zip(p_positions, p_holm_vals):
            records[i][(metric, "p_holm")] = p_holm

    for row in records:
        for metric in METRICS:
            row.setdefault((metric, "p_holm"), np.nan)

    return records


# --------------------------------------------------------------------------
# LaTeX formatting helpers
# --------------------------------------------------------------------------
def format_p(p, bold=False):
    if np.isnan(p):
        return "--"
    if p < 1e-3:
        exponent = int(np.floor(np.log10(p)))
        mantissa = p / (10**exponent)
        inner = f"{mantissa:.2f} \\times 10^{{{exponent}}}"
        if bold:
            inner = f"\\boldsymbol{{{inner}}}"
        return f"${inner}$"
    else:
        text = f"{p:.3g}"
        return f"\\textbf{{{text}}}" if bold else text


# --------------------------------------------------------------------------
# LaTeX table assembly
# --------------------------------------------------------------------------
def build_latex(records):
    n_metrics = len(METRICS)
    colspec = "|c|c|c|" + "c" * n_metrics + "|"
    n_cols = 3 + n_metrics

    lines = []
    lines.append("\\begin{table}")
    lines.append("    \\centering")
    lines.append("    \\footnotesize")
    lines.append("    \\begin{adjustbox}{max width=\\textwidth}")
    lines.append(f"    \\begin{{tabular}}{{{colspec}}}")
    lines.append("        \\hline")

    # Single header row: identifying columns + one column per metric
    header = [
        "\\makecell{Run \\\\ quantity}",
        "\\makecell{Dataset \\\\ samples}",
        "Dataset",
    ]
    for metric in METRICS:
        header.append(METRIC_LABELS[metric])
    lines.append("        " + " & ".join(header) + " \\\\")
    lines.append("        \\hline")

    # Body rows
    idx = 0
    for s in SAMPLE_COUNTS:
        for n_total in DATASET_SIZES:
            for pos_in_block, family in enumerate(FAMILIES):
                row = records[idx]
                idx += 1

                cells = []

                # Run quantity: multirow across the whole (len(DATASET_SIZES)*len(FAMILIES)) block
                if n_total == DATASET_SIZES[0] and pos_in_block == 0:
                    span = len(DATASET_SIZES) * len(FAMILIES)
                    cells.append(f"\\multirow{{{span}}}{{*}}{{{s}}}")
                else:
                    cells.append("")

                # Dataset samples: multirow across the len(FAMILIES) rows for this n_total
                if pos_in_block == 0:
                    cells.append(f"\\multirow{{{len(FAMILIES)}}}{{*}}{{{n_total}}}")
                else:
                    cells.append("")

                # Dataset: one per row
                cells.append(family)

                # Metric value columns -- p_holm only
                for metric in METRICS:
                    p_holm = row[(metric, "p_holm")]
                    bold = (not np.isnan(p_holm)) and (p_holm < ALPHA)
                    cells.append(format_p(p_holm, bold=bold))

                lines.append("        " + " & ".join(cells) + " \\\\")

                # Horizontal rules at group boundaries
                if pos_in_block == len(FAMILIES) - 1:
                    if n_total == DATASET_SIZES[-1]:
                        lines.append("        \\hline")
                    else:
                        lines.append(f"        \\cline{{2-{n_cols}}}")
                else:
                    lines.append(f"        \\cline{{3-{n_cols}}}")

    lines.append("    \\end{tabular}")
    lines.append("    \\end{adjustbox}")
    lines.append(
        "    \\caption{Holm-adjusted Shapiro-Wilk $p$-values ($p_{\\mathrm{holm}}$) for "
        "microTEDAclus evaluation metrics measured at the batch after the middle one, "
        "across dataset families, dataset sizes, and run counts. Correction is applied "
        "jointly across all 48 tests. Bold indicates rejection of normality at "
        "$\\alpha=0.05$.}"
    )
    lines.append("    \\label{tab:normality_shapiro}")
    lines.append("\\end{table}")

    return "\n".join(lines)


def main():
    records, warnings = collect_records()
    records = add_holm_correction(records)
    latex = build_latex(records)

    with open(OUTPUT_PATH, "w") as f:
        f.write(latex + "\n")

    print(f"Wrote {OUTPUT_PATH}")

    if warnings:
        print("\nWarnings:")
        for w in warnings:
            print(" -", w)


if __name__ == "__main__":
    main()
