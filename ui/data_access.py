"""
Reads the cleaned encounter table from Postgres for the dashboard.
Cached via st.cache_data so repeated filter changes in the UI don't
re-hit the database on every rerun.
"""
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine

from etl.config import CLEANED_TABLE_NAME, DATABASE_URL


@st.cache_data(ttl=600)
def load_cleaned_data() -> pd.DataFrame:
    engine = create_engine(DATABASE_URL)
    return pd.read_sql_table(CLEANED_TABLE_NAME, engine)
