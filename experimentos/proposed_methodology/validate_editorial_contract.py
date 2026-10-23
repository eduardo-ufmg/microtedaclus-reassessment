#!/usr/bin/env python3
"""Validate source and generated-artifact editorial contracts."""

import os
import re
import sys

from editorial_config import (
    COMPARISONS,
    DATASET_ORDER,
    DATASET_SHORT_NAMES,
    DATASET_SPECS,
    MODEL_ORDER,
)
from pfc_shared.figures import FOCUS_COLOR, MODEL_COLORS

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPOSITORY_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
THESIS_DIR = os.path.join(REPOSITORY_ROOT, "monografia")

EXPECTED_TABLES = ("quality", "cost", "microclusters", "scalability")


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def validate_configuration() -> None:
    if len(MODEL_ORDER) != len(set(MODEL_ORDER)):
        fail("MODEL_ORDER contains duplicate model names")
    if len(DATASET_ORDER) != len(set(DATASET_ORDER)):
        fail("DATASET_ORDER contains duplicate dataset names")
    if set(DATASET_SHORT_NAMES) != set(DATASET_ORDER):
        fail("DATASET_SHORT_NAMES does not cover exactly DATASET_ORDER")
    colors = list(MODEL_COLORS.values()) + [FOCUS_COLOR]
    if len(set(colors)) != len(colors):
        fail("the figure colors are not distinct")

    for comparison in COMPARISONS:
        models = (comparison.focus, *comparison.others)
        if len(set(models)) != len(models):
            fail(f"comparison {comparison.name} repeats a model")
        unknown = set(models) - set(MODEL_ORDER)
        if unknown:
            fail(f"comparison {comparison.name} has unknown models: {unknown}")
        uncolored = set(comparison.others) - set(MODEL_COLORS)
        if uncolored:
            fail(
                f"comparison {comparison.name} has models without a color: {uncolored}"
            )


def validate_generated_tables() -> None:
    for comparison in COMPARISONS:
        table_dir = os.path.join(
            THESIS_DIR, comparison.chapter, "artifacts", comparison.name
        )
        if not os.path.isdir(table_dir):
            fail(f"missing generated table directory: {table_dir}")

        for table in EXPECTED_TABLES:
            table_path = os.path.join(table_dir, f"{table}.tex")
            if not os.path.isfile(table_path):
                fail(f"missing generated table: {table_path}")

            with open(table_path, encoding="utf-8") as table_file:
                content = table_file.read()

            expected_label = f"\\label{{tab:{comparison.label}_{table}}}"
            if expected_label not in content:
                fail(f"missing or incorrect table label: {table_path}")
            if "\\caption" not in content or "\\begin{tabular}" not in content:
                fail(f"incomplete table structure: {table_path}")
            if content.index("\\caption") > content.index("\\begin{tabular}"):
                fail(f"table caption is not above tabular content: {table_path}")


def validate_result_content() -> None:
    allowed_ids = set(DATASET_SPECS) | {
        "SCL-1",
        "SCL-2",
        "SCL-4",
        "SCL-8",
        "SCL-16",
        "SCL-32",
        "SCL-64",
    }
    params_dir = os.path.join(SCRIPT_DIR, "results_params")
    for filename in os.listdir(params_dir):
        if not filename.endswith(".csv"):
            continue
        path = os.path.join(params_dir, filename)
        with open(path, encoding="utf-8") as params_file:
            header = params_file.readline().rstrip("\n").split(",")
            if "dataset" not in header:
                continue
            dataset_index = header.index("dataset")
            for line_number, line in enumerate(params_file, start=2):
                fields = line.rstrip("\n").split(",")
                if (
                    dataset_index < len(fields)
                    and fields[dataset_index] not in allowed_ids
                ):
                    fail(
                        f"legacy or unknown dataset ID {fields[dataset_index]!r} "
                        f"in {path}:{line_number}"
                    )


def validate_thesis_inputs() -> None:
    thesis_source = os.path.join(
        THESIS_DIR, "MethodologyReassessment", "MethodologyReassessment.tex"
    )
    with open(thesis_source, encoding="utf-8") as source_file:
        source = source_file.read()

    inputs = re.findall(r"\\input\{([^}]+)\}", source)
    for relative_path in inputs:
        input_path = os.path.join(THESIS_DIR, relative_path)
        if not os.path.isfile(input_path):
            fail(f"missing thesis input: {input_path}")


def main() -> None:
    validate_configuration()
    validate_result_content()
    validate_generated_tables()
    validate_thesis_inputs()
    print("editorial-contract: ok")


if __name__ == "__main__":
    main()
