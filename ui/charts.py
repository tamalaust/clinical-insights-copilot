"""
Pure data-shaping functions for the dashboard charts: cleaned DataFrame
in, chart-ready DataFrame out. Kept independent of Streamlit/Plotly
rendering so this is testable the same way etl/clean.py is — see
tests/test_charts.py.
"""
from __future__ import annotations

import pandas as pd

# Natural ordering for the age bucket labels, since sorting them
# alphabetically puts "[10-20)" before "[0-10)".
AGE_ORDER = [
    "[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)",
    "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)",
]


def readmission_rate_by_age(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns one row per age bucket with the share of encounters that
    were readmitted (readmitted != 'NO'), as a percentage.
    """
    result = (
        df.assign(was_readmitted=df["readmitted"] != "NO")
        .groupby("age", observed=True)["was_readmitted"]
        .mean()
        .reindex(AGE_ORDER)
        .dropna()
        .mul(100)
        .reset_index()
        .rename(columns={"was_readmitted": "readmission_rate_pct"})
    )
    return result


def time_in_hospital_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Returns one row per distinct time_in_hospital value with its count."""
    return (
        df["time_in_hospital"]
        .value_counts()
        .sort_index()
        .reset_index()
        .rename(columns={"count": "encounter_count"})
    )


def admission_type_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Returns encounter counts per admission type description."""
    return (
        df["admission_type_id_description"]
        .value_counts()
        .reset_index()
        .rename(columns={"count": "encounter_count"})
    )


def avg_medications_by_age(df: pd.DataFrame) -> pd.DataFrame:
    """Returns average num_medications per age bucket."""
    result = (
        df.groupby("age", observed=True)["num_medications"]
        .mean()
        .reindex(AGE_ORDER)
        .dropna()
        .reset_index()
        .rename(columns={"num_medications": "avg_num_medications"})
    )
    return result
