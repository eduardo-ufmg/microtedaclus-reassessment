import math
import os
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

# Process files in metrics_summary/ to create LaTeX tables for beamer.
# Cell format is <median> [<q1>, <q3>]. "Best" per dataset row is decided
# by Mann-Whitney U tests against the top median (not by median comparison
# alone), Holm-Bonferroni corrected across every pairwise test performed
# for a given metric table (i.e. across all datasets x all model pairs
# compared while building that one table). A model is bolded if it *is*
# the top-median model, or if it is not statistically distinguishable
# from it after correction.
#
# Also builds one hyperparameter table per model from
# metrics_summary/params/{model}.csv (produced by aggregate_params.py) --
# no significance testing applies there (different models have different,
# non-comparable parameter spaces).

INPUT_DIR = "./metrics_summary"
PARAMS_INPUT_DIR = "./metrics_summary/params"
PER_RUN_DIR = "./aggregated_per_run"
OUTPUT_DIR = "./latex/tables"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_ORDER = [
    "CluStream",
    "DenStream",
    "MicroTEDAClus",
    "CEDAS",
    "TEDACloud",
    "AutoCloud",
    "StreamKM",
]
SCL_PATTERN = re.compile(r"^SCL[-_](\d+)$")
ALPHA = 0.05


@dataclass(frozen=True)
class MetricSpec:
    key: str
    display: str
    best: str  # "min" or "max"
    scale: float
    unit: str | None
    round_to_int: bool = False


METRICS: list[MetricSpec] = [
    MetricSpec("ari_diff_max_last", "ARI diff (max-last)", "min", 1.0, None),
    MetricSpec("ari_median_all", "ARI median (all)", "max", 1.0, None),
    MetricSpec("ari_median_first_10", "ARI median (first 10)", "max", 1.0, None),
    MetricSpec("ari_median_last_10", "ARI median (last 10)", "max", 1.0, None),
    MetricSpec("memory_bytes_median", "Memory", "min", 1 / (1024**2), "MB"),
    MetricSpec(
        "memory_bytes_iqr_width",
        "Memory (intra-run IQR width)",
        "min",
        1 / (1024**2),
        "MB",
    ),
    MetricSpec("memory_bytes_max", "Memory (max)", "min", 1 / (1024**2), "MB"),
    MetricSpec("time_fit_ns_median", "Fit time", "min", 1e-6, "ms"),
    MetricSpec(
        "time_fit_ns_iqr_width", "Fit time (intra-run IQR width)", "min", 1e-6, "ms"
    ),
    MetricSpec("time_predict_ns_median", "Predict time", "min", 1e-6, "ms"),
    MetricSpec(
        "time_predict_ns_iqr_width",
        "Predict time (intra-run IQR width)",
        "min",
        1e-6,
        "ms",
    ),
    MetricSpec(
        "microcluster_count_median", "Microcluster count", "min", 1.0, None, True
    ),
    MetricSpec(
        "microcluster_count_iqr_width",
        "Microcluster count (intra-run IQR width)",
        "min",
        1.0,
        None,
        True,
    ),
    MetricSpec(
        "microcluster_count_max", "Microcluster count (max)", "min", 1.0, None, True
    ),
]


def escape_latex(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    return text


def _num_pattern() -> str:
    return r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"


def parse_cell(cell_value: str) -> tuple[float | None, float | None, float | None]:
    """Parses a '<median> [<q1>, <q3>]' cell. Returns (median, q1, q3)."""
    if cell_value is None:
        return None, None, None

    text = str(cell_value).strip()
    if text == "" or text.lower() == "nan":
        return None, None, None

    n = _num_pattern()
    match = re.match(rf"^\s*({n})\s*\[\s*({n})\s*,\s*({n})\s*\]\s*$", text)
    if match:
        return float(match.group(1)), float(match.group(2)), float(match.group(3))

    try:
        value = float(text)
        return value, value, value
    except ValueError:
        print(f"Warning: could not parse cell value '{text}'.")
        return None, None, None


def format_cell(
    median_val: float | None,
    q1_val: float | None,
    q3_val: float | None,
    round_to_int: bool = False,
) -> str:
    if median_val is None or q1_val is None or q3_val is None:
        return "{NaN}"
    if math.isnan(median_val) or math.isnan(q1_val) or math.isnan(q3_val):
        return "{NaN}"
    if round_to_int:
        median_val = float(round(median_val))
        q1_val = float(round(q1_val))
        q3_val = float(round(q3_val))
        return f"{median_val:.0f} [{q1_val:.0f}, {q3_val:.0f}]"
    return f"{median_val:.3f} [{q1_val:.3f}, {q3_val:.3f}]"


def is_best(value: float, best_value: float) -> bool:
    return math.isclose(value, best_value, rel_tol=1e-9, abs_tol=1e-12)


def dataset_sort_key(dataset_name: str) -> tuple[int, str, int]:
    match = SCL_PATTERN.match(dataset_name)
    if match:
        return 1, "SCL-", int(match.group(1))
    return 0, dataset_name, 0


def build_caption(spec: MetricSpec) -> str:
    if spec.unit:
        return f"{spec.display} ({spec.unit})"
    return spec.display


def sort_by_dataset(df: pd.DataFrame) -> pd.DataFrame:
    dataset_values = df["dataset"].fillna("").astype(str).tolist()
    sort_keys = [dataset_sort_key(value) for value in dataset_values]
    df = df.copy()
    df["_sort_group"] = [key[0] for key in sort_keys]
    df["_sort_alpha"] = [key[1] for key in sort_keys]
    df["_sort_num"] = [key[2] for key in sort_keys]
    df = df.sort_values(
        ["_sort_group", "_sort_alpha", "_sort_num"], kind="mergesort"
    ).drop(columns=["_sort_group", "_sort_alpha", "_sort_num"])
    return df


# --------------------------------------------------------------------------
# Significance testing (Mann-Whitney U, Holm-Bonferroni corrected across
# every pairwise test performed while building one metric table)
# --------------------------------------------------------------------------
def load_run_values(dataset: str, model: str, metric_key: str) -> np.ndarray:
    """Raw per-run values for (dataset, model, metric_key), read from
    aggregated_per_run/{dataset}_{model}.csv — i.e. the per-run point
    statistics that metrics_summary/*.csv was itself computed from."""
    path = os.path.join(PER_RUN_DIR, f"{dataset}_{model}.csv")
    if not os.path.exists(path):
        return np.array([])
    try:
        df = pd.read_csv(path)
    except Exception:
        return np.array([])
    if metric_key not in df.columns:
        return np.array([])
    return pd.to_numeric(df[metric_key], errors="coerce").dropna().to_numpy()


def holm_bonferroni(p_values: list[float], alpha: float = ALPHA) -> np.ndarray:
    """Holm step-down procedure. Returns a boolean array (True = reject
    H0 = statistically significant difference), aligned to input order."""
    m = len(p_values)
    if m == 0:
        return np.array([], dtype=bool)

    p_arr = np.asarray(p_values, dtype=float)
    order = np.argsort(p_arr)
    sorted_p = p_arr[order]

    reject_sorted = np.zeros(m, dtype=bool)
    for i, p in enumerate(sorted_p):
        threshold = alpha / (m - i)
        if p <= threshold:
            reject_sorted[i] = True
        else:
            break  # Holm step-down stops at the first non-rejection

    reject = np.zeros(m, dtype=bool)
    reject[order] = reject_sorted
    return reject


def build_table(metric_path: str, spec: MetricSpec) -> str:
    df = pd.read_csv(metric_path, dtype=str)
    if "dataset" not in df.columns:
        raise ValueError(f"Missing 'dataset' column in {metric_path}.")

    for model in MODEL_ORDER:
        if model not in df.columns:
            df[model] = "NaN"

    df = df[["dataset"] + MODEL_ORDER]
    df = sort_by_dataset(df)

    # --- Phase 1: parse cells, find the exact-tied-best model(s) per row,
    # and schedule pairwise Mann-Whitney tests of every other model
    # against a representative best model. ---
    rows_data = []  # per-row: dataset name, raw dataset value, row_values, exact_tied
    pending_tests = []  # list of dicts: row_idx, model, p_value

    for row_idx, (_, row) in enumerate(df.iterrows()):
        raw_dataset = str(row["dataset"])
        row_values: dict[str, tuple[float | None, float | None, float | None]] = {}
        for model in MODEL_ORDER:
            median_val, q1_val, q3_val = parse_cell(row[model])
            if median_val is not None:
                median_val *= spec.scale
            if q1_val is not None:
                q1_val *= spec.scale
            if q3_val is not None:
                q3_val *= spec.scale
            row_values[model] = (median_val, q1_val, q3_val)

        numeric_values = {
            model: median
            for model, (median, _, _) in row_values.items()
            if median is not None and not math.isnan(median)
        }

        exact_tied: set[str] = set()
        if numeric_values:
            best_value = (
                max(numeric_values.values())
                if spec.best == "max"
                else min(numeric_values.values())
            )
            exact_tied = {
                model
                for model, median in numeric_values.items()
                if is_best(median, best_value)
            }

        rows_data.append(
            {
                "dataset_name": escape_latex(raw_dataset),
                "raw_dataset": raw_dataset,
                "row_values": row_values,
                "exact_tied": exact_tied,
            }
        )

        if len(numeric_values) < 2:
            continue

        ref_model = sorted(exact_tied, key=MODEL_ORDER.index)[0]
        ref_vals = load_run_values(raw_dataset, ref_model, spec.key)

        for model in numeric_values:
            if model in exact_tied:
                continue
            other_vals = load_run_values(raw_dataset, model, spec.key)
            if len(ref_vals) == 0 or len(other_vals) == 0:
                print(
                    f"Warning: no per-run data for {raw_dataset}/{model}/{spec.key} "
                    "-- skipping significance test, not bolding."
                )
                continue
            try:
                _, p_value = mannwhitneyu(other_vals, ref_vals, alternative="two-sided")
            except Exception as e:
                print(
                    f"Warning: Mann-Whitney U failed for {raw_dataset}/{model} vs "
                    f"{ref_model} on {spec.key}: {e} -- not bolding."
                )
                continue

            pending_tests.append(
                {"row_idx": row_idx, "model": model, "p_value": p_value}
            )

    # --- Phase 2: Holm-Bonferroni correction across every test performed
    # for this table (all datasets x all pairwise comparisons together). ---
    reject = holm_bonferroni([t["p_value"] for t in pending_tests], ALPHA)
    tied_with_best: dict[int, set[str]] = {}
    for test, rejected in zip(pending_tests, reject):
        if not rejected:  # not significantly different from best -> tied
            tied_with_best.setdefault(test["row_idx"], set()).add(test["model"])

    # --- Phase 3: assemble LaTeX rows using the corrected bold set. ---
    rows_tex: list[str] = []
    for row_idx, row_info in enumerate(rows_data):
        best_models = row_info["exact_tied"] | tied_with_best.get(row_idx, set())

        cell_texts: list[str] = []
        for model in MODEL_ORDER:
            median_val, q1_val, q3_val = row_info["row_values"][model]
            cell_tex = format_cell(median_val, q1_val, q3_val, spec.round_to_int)
            if cell_tex != "{NaN}" and model in best_models:
                cell_tex = f"\\bfseries {cell_tex}"
            cell_texts.append(cell_tex)

        row_line = " & ".join([row_info["dataset_name"]] + cell_texts) + r" \\"
        rows_tex.append(row_line)

    # "median [q1, q3]" cells are plain text, not siunitx-alignable numbers
    # the way "mean \pm std" was, so plain left-aligned columns are used
    # instead of the previous dynamic-width S[table-format=...] columns.
    column_spec = "l" + "l" * len(MODEL_ORDER)
    caption = (
        build_caption(spec)
        + f" (bold: not significantly different from the best, Mann-Whitney "
        + f"U, Holm-Bonferroni corrected, $\\alpha={ALPHA}$)"
    )
    label = f"tab:{spec.key}"

    header_models = " & ".join([f"{{{m}}}" for m in MODEL_ORDER])

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{{column_spec}}}",
        r"\hline",
        f"Dataset & {header_models} \\\\",
        r"\hline",
        *rows_tex,
        r"\hline",
        r"\end{tabular}",
        r"}",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\end{table}",
    ]

    return "\n".join(lines) + "\n"


def build_params_table(params_path: str, model_name: str) -> str:
    """Builds a hyperparameter table for one model: rows = dataset,
    columns = that model's tuned parameters. Cells are already formatted
    by aggregate_params.py (median [Q1,Q3] for numeric, mode (count/total)
    for categorical) -- no re-parsing or significance testing applies
    here (parameters are not comparable across models)."""
    df = pd.read_csv(params_path, dtype=str)
    if "dataset" not in df.columns:
        raise ValueError(f"Missing 'dataset' column in {params_path}.")

    df = sort_by_dataset(df)
    param_cols = [c for c in df.columns if c != "dataset"]

    rows_tex: list[str] = []
    for _, row in df.iterrows():
        dataset_name = escape_latex(str(row["dataset"]))
        cell_texts = []
        for col in param_cols:
            val = row[col]
            cell_texts.append("{NaN}" if pd.isna(val) else escape_latex(str(val)))
        row_line = " & ".join([dataset_name] + cell_texts) + r" \\"
        rows_tex.append(row_line)

    column_spec = "l" + "l" * len(param_cols)
    header_params = " & ".join([f"{{{escape_latex(p)}}}" for p in param_cols])
    caption = f"Tuned hyperparameters --- {escape_latex(model_name)} (median [Q1, Q3], or mode (count/total) for categorical parameters)"
    label = f"tab:params_{model_name}"

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{{column_spec}}}",
        r"\hline",
        f"Dataset & {header_params} \\\\",
        r"\hline",
        *rows_tex,
        r"\hline",
        r"\end{tabular}",
        r"}",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\end{table}",
    ]

    return "\n".join(lines) + "\n"


def main() -> None:
    generated = []

    for spec in METRICS:
        metric_path = os.path.join(INPUT_DIR, f"{spec.key}.csv")
        if not os.path.exists(metric_path):
            print(f"Skipping {spec.key}: file not found.")
            continue

        table_tex = build_table(metric_path, spec)
        output_path = os.path.join(OUTPUT_DIR, f"{spec.key}.tex")
        with open(output_path, "w", encoding="utf-8") as handle:
            handle.write(table_tex)
        generated.append(output_path)

    if os.path.isdir(PARAMS_INPUT_DIR):
        for model in MODEL_ORDER:
            params_path = os.path.join(PARAMS_INPUT_DIR, f"{model}.csv")
            if not os.path.exists(params_path):
                print(f"Skipping params for {model}: file not found.")
                continue

            table_tex = build_params_table(params_path, model)
            output_path = os.path.join(OUTPUT_DIR, f"params_{model}.tex")
            with open(output_path, "w", encoding="utf-8") as handle:
                handle.write(table_tex)
            generated.append(output_path)
    else:
        print(f"Skipping parameter tables: {PARAMS_INPUT_DIR} not found.")

    if generated:
        print("Generated LaTeX tables:")
        for path in generated:
            print(f"- {path}")
    else:
        print("No LaTeX tables generated.")


if __name__ == "__main__":
    main()
