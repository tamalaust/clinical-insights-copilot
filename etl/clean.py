"""
Pure data-cleaning functions: raw DataFrame in, cleaned DataFrame out.

Kept independent of any DB/IO so it's trivially unit-testable — see
tests/test_clean.py.
"""
from __future__ import annotations

import pandas as pd

from etl.config import IQR_MULTIPLIER, NUMERIC_COLUMNS_FOR_OUTLIER_CHECK, IDS_MAPPING_PATH
from etl.mappings import load_id_mappings, apply_id_mappings

# Values the raw dataset uses to represent "missing" that pandas won't
# recognize as NaN out of the box.
MISSING_VALUE_TOKENS = ["?", "None", "Unknown/Invalid", ""]


def load_raw(path) -> pd.DataFrame:
    """Read the raw CSV, mapping known missing-value tokens to NaN."""
    return pd.read_csv(path, na_values=MISSING_VALUE_TOKENS)


def normalize_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize missing-value representation across the frame.
    Currently a light pass since load_raw already maps known tokens to NaN
    on read; this exists as an explicit step so future rules land in one
    place rather than being scattered through the pipeline.
    """
    return df.copy()


def fix_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Coerce numeric columns to numeric dtype, catching values that slipped
    through as strings (common with encounter IDs, mixed CSV exports).
    Non-numeric-coercible values become NaN rather than raising.
    """
    df = df.copy()
    for col in NUMERIC_COLUMNS_FOR_OUTLIER_CHECK:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def flag_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flag outliers per numeric column using the IQR (Tukey fence) method.
    Adds one boolean column per checked column: `<col>_is_outlier`.

    Rows are NEVER dropped here — flagging preserves every row so
    downstream consumers (dashboard, agent) can decide how to treat
    flagged values instead of having them silently disappear.
    """
    df = df.copy()
    for col in NUMERIC_COLUMNS_FOR_OUTLIER_CHECK:
        if col not in df.columns:
            continue
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - IQR_MULTIPLIER * iqr
        upper = q3 + IQR_MULTIPLIER * iqr
        df[f"{col}_is_outlier"] = (df[col] < lower) | (df[col] > upper)
    return df


def decode_id_columns(df: pd.DataFrame, mappings_path=IDS_MAPPING_PATH) -> pd.DataFrame:
    """
    Join human-readable descriptions onto admission_type_id,
    discharge_disposition_id, and admission_source_id. Original ID
    columns are kept; `<col>_description` columns are added.
    """
    mappings = load_id_mappings(mappings_path)
    return apply_id_mappings(df, mappings)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Full cleaning pipeline: raw -> normalized -> typed -> decoded -> outlier-flagged."""
    df = normalize_missing_values(df)
    df = fix_dtypes(df)
    df = decode_id_columns(df)
    df = flag_outliers(df)
    return df
