#!/usr/bin/env python3
"""Validate source and generated-artifact editorial contracts."""

import os
import re
import sys

from editorial_config import DATASET_ORDER, MODEL_COLORS, MODEL_ORDER, SUBSETS

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPOSITORY_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
THESIS_DIR = os.path.join(REPOSITORY_ROOT, "monografia")
ARTIFACT_DIR = os.path.join(THESIS_DIR, "MethodologyReassessment", "artifacts")

EXPECTED_SUBSETS = set(SUBSETS)
EXPECTED_TABLES = {
    "ari_diff",
    "ari_fifth",
    "ari_overall",
    "ari_penultimate",
    "mc_max",
    "mc_penultimate",
    "memory_max_bytes",
    "memory_penultimate_bytes",
    "time_fit_cum_ms",
    "time_predict_cum_ms",
}


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def validate_configuration() -> None:
    if len(MODEL_ORDER) != len(set(MODEL_ORDER)):
        fail("MODEL_ORDER contains duplicate model names")
    if len(DATASET_ORDER) != len(set(DATASET_ORDER)):
        fail("DATASET_ORDER contains duplicate dataset names")
    if set(MODEL_COLORS) != set(MODEL_ORDER):
        fail("MODEL_COLORS does not cover exactly MODEL_ORDER")
    if len(set(MODEL_COLORS.values())) != len(MODEL_COLORS):
        fail("MODEL_COLORS contains duplicate colors")

    for subset_name, models in SUBSETS.items():
        if models != [model for model in MODEL_ORDER if model in models]:
            fail(f"subset {subset_name} is not an ordered projection of MODEL_ORDER")


def validate_generated_tables() -> None:
    for subset_name in EXPECTED_SUBSETS:
        table_dir = os.path.join(ARTIFACT_DIR, subset_name, "tables")
        if not os.path.isdir(table_dir):
            fail(f"missing generated table directory: {table_dir}")

        for metric in EXPECTED_TABLES:
            table_path = os.path.join(table_dir, f"{metric}.tex")
            if not os.path.isfile(table_path):
                fail(f"missing generated table: {table_path}")

            with open(table_path, encoding="utf-8") as table_file:
                content = table_file.read()

            expected_label = f"\\label{{tab:{subset_name}_{metric}}}"
            if expected_label not in content:
                fail(f"missing or incorrect table label: {table_path}")
            if "\\caption{" not in content or "\\begin{tabular}" not in content:
                fail(f"incomplete table structure: {table_path}")
            if content.index("\\caption{") > content.index("\\begin{tabular}"):
                fail(f"table caption is not above tabular content: {table_path}")


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
    validate_generated_tables()
    validate_thesis_inputs()
    print("editorial-contract: ok")


if __name__ == "__main__":
    main()
