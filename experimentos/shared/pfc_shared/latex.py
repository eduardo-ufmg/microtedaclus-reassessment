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


def focus_rows(compared: pd.DataFrame, focus: str, others, label, text) -> list:
    """Rows of a panel of the comparison of a model in focus with each other model.

    `compared` is the result of `pfc_shared.statistics.compare_with_focus`. The
    rows are the model in focus, its count of verdicts, and every other model,
    marked by how the model in focus compares with it. `label` gives the name of
    a model in the table and `text` the text of a median.
    """
    of_focus = compared[compared.model == focus].set_index("dataset")
    counts = {
        dataset: "/".join(
            str((of_dataset.verdict == verdict).sum())
            for verdict in ("better", "same", "worse")
        )
        for dataset, of_dataset in compared[compared.model != focus].groupby("dataset")
    }
    rows = [
        (label(focus), of_focus["median"].to_dict(), text),
        ("Better/same/worse", counts, str),
    ]
    for model in others:
        of_model = compared[compared.model == model].set_index("dataset")
        rows.append(
            (
                label(model),
                {
                    dataset: marked(text(row["median"]), row["verdict"])
                    for dataset, row in of_model.iterrows()
                },
                str,
            )
        )
    return rows


def write_table(
    path: Path,
    caption: str,
    label: str,
    column_spec: str,
    lines: list[str],
    short: str | None = None,
):
    """Write a table at its natural size, reduced only if it exceeds the text block.

    `short`, if given, is the entry of the table in the list of tables.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    body = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        rf"\caption[{short}]{{{caption}}}" if short else rf"\caption{{{caption}}}",
        r"\adjustbox{max width=\textwidth, max totalheight=0.82\textheight}{%",
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
    path: Path,
    caption: str,
    label: str,
    columns: dict,
    panels,
    first_column="Model",
    short: str | None = None,
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
    write_table(path, caption, label, "l " + "c " * len(columns), lines, short)
