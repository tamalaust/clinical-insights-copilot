"""
Loads IDS_mapping.csv, which decodes admission_type_id,
discharge_disposition_id, and admission_source_id into human-readable
descriptions.

The file is 3 sections concatenated together, each starting with its own
header row (e.g. "admission_type_id,description") and separated by a
blank line. This parses by header, not by fixed line numbers, so it
tolerates the sections being reordered or padded differently.
"""
from __future__ import annotations

import pandas as pd

SECTION_HEADERS = [
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
]


def load_id_mappings(path) -> dict[str, dict[int, str]]:
    """
    Returns e.g. {
        "admission_type_id": {1: "Emergency", 2: "Urgent", ...},
        "discharge_disposition_id": {1: "Discharged to home", ...},
        "admission_source_id": {...},
    }
    """
    with open(path) as f:
        raw_lines = [line.rstrip("\n").rstrip("\r") for line in f]

    # find the line index where each section's header starts
    section_starts = {}
    for i, line in enumerate(raw_lines):
        first_col = line.split(",")[0]
        if first_col in SECTION_HEADERS:
            section_starts[first_col] = i

    # sort sections by their start line so we know where each one ends
    ordered = sorted(section_starts.items(), key=lambda kv: kv[1])

    mappings: dict[str, dict[int, str]] = {}
    for idx, (col_name, start_line) in enumerate(ordered):
        end_line = ordered[idx + 1][1] if idx + 1 < len(ordered) else len(raw_lines)
        # +1 to skip the header row itself; blank lines before the next
        # section are dropped by pandas' default skip_blank_lines
        section_lines = raw_lines[start_line:end_line]
        section_csv = "\n".join(section_lines)

        from io import StringIO

        section_df = pd.read_csv(StringIO(section_csv))
        section_df = section_df.dropna(subset=[col_name])
        mappings[col_name] = dict(
            zip(section_df[col_name].astype(int), section_df["description"])
        )

    return mappings


def apply_id_mappings(df: pd.DataFrame, mappings: dict[str, dict[int, str]]) -> pd.DataFrame:
    """
    Adds a `<col>_description` column for each mapped ID column found in df.
    Original ID columns are kept — this is additive, not a replacement.
    """
    df = df.copy()
    for col_name, lookup in mappings.items():
        if col_name in df.columns:
            df[f"{col_name}_description"] = df[col_name].map(lookup)
    return df
