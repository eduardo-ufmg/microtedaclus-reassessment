"""LaTeX tables of the thesis: one writer for every generated table."""

from pathlib import Path

import pandas as pd


def marked(text: str, verdict: str) -> str:
    """Emphasis of a value: bold if better than its reference, red if worse."""
    if verdict == "better":
        return rf"\textbf{{{text}}}"
    if verdict == "worse":
        return rf"\textcolor{{red}}{{{text}}}"
    return text


def write_table(
    path: Path, caption: str, label: str, column_spec: str, lines: list[str]
):
    """Write a table that fits the text block in both directions."""
    path.parent.mkdir(parents=True, exist_ok=True)
    body = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{caption}}}",
        r"\adjustbox{width=\textwidth, totalheight=0.82\textheight, keepaspectratio}{%",
        rf"\begin{{tabular}}{{{column_spec}}}",
        r"\toprule",
        *lines,
        r"\bottomrule",
        r"\end{tabular}",
        "}",
        rf"\label{{{label}}}",
        r"\end{table}",
    ]
    path.write_text("\n".join(body) + "\n")


def panel_table(
    path: Path, caption: str, label: str, columns: dict, panels, first_column="Model"
):
    """Panels of rows with one value per column.

    `columns` maps the key of each column to its header. `panels` is a list of
    (title, rows), and a row is (label, values, fmt), with `values` indexed by
    column key and `fmt` the function that formats one value.
    """
    lines = [f"{first_column} & " + " & ".join(columns.values()) + r" \\"]
    for title, rows in panels:
        lines.append(r"\midrule")
        lines.append(
            rf"\multicolumn{{{1 + len(columns)}}}{{l}}{{\textit{{{title}}}}} \\"
        )
        for row_label, values, fmt in rows:
            cells = [
                fmt(values[key]) if key in values and not pd.isna(values[key]) else "-"
                for key in columns
            ]
            lines.append(" & ".join([row_label, *cells]) + r" \\")
    write_table(path, caption, label, "l " + "c " * len(columns), lines)
