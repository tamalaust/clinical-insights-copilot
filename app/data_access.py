"""
The actual Postgres read, with no dependency on Streamlit or FastAPI.
Both app/tools.py (backend) and ui/data_access.py (UI, which wraps this
with st.cache_data) use this — keeping the read logic itself framework-
agnostic, since app/ shouldn't need streamlit installed just to serve
API requests.
"""
import pandas as pd
from sqlalchemy import create_engine

from etl.config import CLEANED_TABLE_NAME, DATABASE_URL


def read_cleaned_table() -> pd.DataFrame:
    engine = create_engine(DATABASE_URL)
    return pd.read_sql_table(CLEANED_TABLE_NAME, engine)
