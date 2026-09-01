"""
Load step: writes a cleaned DataFrame to Postgres.

Kept separate from clean.py so the cleaning logic can be unit-tested
without a live database, and this module can be swapped/mocked in tests
that only care about the write behavior.
"""
from __future__ import annotations

import pandas as pd
from sqlalchemy import create_engine

from etl.config import CLEANED_TABLE_NAME, DATABASE_URL


def get_engine():
    return create_engine(DATABASE_URL)


def write_cleaned_table(df: pd.DataFrame, table_name: str = CLEANED_TABLE_NAME) -> None:
    """
    Replace the cleaned table with the given DataFrame.

    if_exists="replace" is intentional: ETL is a once/on-demand full
    re-run per the project's design, not an incremental append. Revisit
    if incremental loads become a requirement.
    """
    engine = get_engine()
    df.to_sql(table_name, engine, if_exists="replace", index=False)
