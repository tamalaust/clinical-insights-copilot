"""
Central config for the ETL pipeline. Reads from environment variables so
local dev, CI, and prod can all point at different Postgres instances
without code changes.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # reads .env in the current working directory, if present


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if value is None:
        raise RuntimeError(
            f"Missing required environment variable: {name}. "
            f"Copy .env.example to .env and set it."
        )
    return value


RAW_DATA_PATH = Path(_require_env("RAW_DATA_PATH"))
IDS_MAPPING_PATH = Path(_require_env("IDS_MAPPING_PATH"))
DATABASE_URL = _require_env("DATABASE_URL")
CLEANED_TABLE_NAME = _require_env("CLEANED_TABLE_NAME")

# Columns treated as numeric for outlier flagging. Adjust to match whichever
# dataset you're actually loading (defaults assume the UCI Diabetes 130-US
# hospitals schema).
NUMERIC_COLUMNS_FOR_OUTLIER_CHECK = [
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
]

# IQR multiplier for outlier flagging (1.5 = standard Tukey fence)
IQR_MULTIPLIER = 1.5