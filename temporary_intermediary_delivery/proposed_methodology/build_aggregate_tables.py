import os
import re
import math
from dataclasses import dataclass

import pandas as pd

# Process files in metrics_summary/ to create LaTeX tables for beamer.
# Cell format is <mean> \pm <std>, with best values bolded per dataset row.

INPUT_DIR = "./metrics_summary"
OUTPUT_DIR = "./latex/tables"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_ORDER = ["CluStream", "DenStream", "MicroTEDAClus"]
SCL_PATTERN = re.compile(r"^SCL[-_](\d+)$")


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
    MetricSpec("ari_mean_all", "ARI mean (all)", "max", 1.0, None),
    MetricSpec("ari_mean_first_10", "ARI mean (first 10)", "max", 1.0, None),
    MetricSpec("ari_mean_last_10", "ARI mean (last 10)", "max", 1.0, None),
    MetricSpec("memory_bytes", "Memory", "min", 1 / (1024**2), "MB"),
    MetricSpec("memory_bytes_max", "Memory (max)", "min", 1 / (1024**2), "MB"),
    MetricSpec("time_fit_ns", "Fit time", "min", 1e-6, "ms"),
    MetricSpec("time_predict_ns", "Predict time", "min", 1e-6, "ms"),
    MetricSpec("microcluster_count", "Microcluster count", "min", 1.0, None, True),
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


def parse_cell(cell_value: str) -> tuple[float | None, float | None]:
    if cell_value is None:
        return None, None

    text = str(cell_value).strip()
    if text == "" or text.lower() == "nan":
        return None, None

    pm_pattern = rf"(?:\\pm|{chr(0x00B1)})"
    match = re.match(
        rf"^\s*({_num_pattern()})\s*{pm_pattern}\s*({_num_pattern()})\s*$", text
    )
    if match:
        return float(match.group(1)), float(match.group(2))

    try:
        return float(text), 0.0
    except ValueError:
        print(f"Warning: could not parse cell value '{text}'.")
        return None, None


def format_cell(
    mean_val: float | None, std_val: float | None, round_to_int: bool = False
) -> str:
    if mean_val is None or std_val is None:
        return "NaN"
    if math.isnan(mean_val) or math.isnan(std_val):
        return "NaN"
    if round_to_int:
        mean_val = float(round(mean_val))
        std_val = float(round(std_val))
        return f"{mean_val:.0f} \\pm {std_val:.0f}"
    return f"{mean_val:.3f} \\pm {std_val:.3f}"


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


def build_table(metric_path: str, spec: MetricSpec) -> str:
    df = pd.read_csv(metric_path, dtype=str)
    if "dataset" not in df.columns:
        raise ValueError(f"Missing 'dataset' column in {metric_path}.")

    for model in MODEL_ORDER:
        if model not in df.columns:
            df[model] = "NaN"

    df = df[["dataset"] + MODEL_ORDER]
    dataset_values = df["dataset"].fillna("").astype(str).tolist()
    sort_keys = [dataset_sort_key(value) for value in dataset_values]
    df["_sort_group"] = [key[0] for key in sort_keys]
    df["_sort_alpha"] = [key[1] for key in sort_keys]
    df["_sort_num"] = [key[2] for key in sort_keys]
    df = df.sort_values(
        ["_sort_group", "_sort_alpha", "_sort_num"], kind="mergesort"
    ).drop(columns=["_sort_group", "_sort_alpha", "_sort_num"])

    rows_tex: list[str] = []
    for _, row in df.iterrows():
        dataset_name = escape_latex(str(row["dataset"]))
        row_values: dict[str, tuple[float | None, float | None]] = {}
        for model in MODEL_ORDER:
            mean_val, std_val = parse_cell(row[model])
            if mean_val is not None:
                mean_val *= spec.scale
            if std_val is not None:
                std_val *= spec.scale
            row_values[model] = (mean_val, std_val)

        numeric_values = {
            model: mean
            for model, (mean, _) in row_values.items()
            if mean is not None and not math.isnan(mean)
        }
        best_models: set[str] = set()
        if numeric_values:
            best_value = (
                max(numeric_values.values())
                if spec.best == "max"
                else min(numeric_values.values())
            )
            best_models = {
                model
                for model, mean in numeric_values.items()
                if is_best(mean, best_value)
            }

        cell_texts: list[str] = []
        for model in MODEL_ORDER:
            mean_val, std_val = row_values[model]
            cell_body = format_cell(mean_val, std_val, spec.round_to_int)
            if cell_body != "NaN":
                cell_tex = f"${cell_body}$"
                if model in best_models:
                    cell_tex = f"{{\\boldmath {cell_tex}}}"
            else:
                cell_tex = "NaN"
            cell_texts.append(cell_tex)

        row_line = " & ".join([dataset_name] + cell_texts) + r" \\"
        rows_tex.append(row_line)

    column_spec = "l" + "c" * len(MODEL_ORDER)
    caption = build_caption(spec)
    label = f"tab:{spec.key}"

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{{column_spec}}}",
        r"\hline",
        "Dataset & " + " & ".join(MODEL_ORDER) + r" \\",
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

    if generated:
        print("Generated LaTeX tables:")
        for path in generated:
            print(f"- {path}")
    else:
        print("No LaTeX tables generated.")


if __name__ == "__main__":
    main()
