"""
Ensures required env vars are set before any test imports etl.config,
so the test suite doesn't depend on a developer's local .env file
existing (config.py intentionally has no fallback defaults).

Uses setdefault so a real .env — if present — still takes priority.
"""
import os

os.environ.setdefault("RAW_DATA_PATH", "data/raw/diabetic_data.csv")
os.environ.setdefault("IDS_MAPPING_PATH", "data/raw/IDS_mapping.csv")
os.environ.setdefault(
    "DATABASE_URL",
    "postgresql+psycopg2://test:test@localhost:5432/clinical_insights_test",
)
os.environ.setdefault("CLEANED_TABLE_NAME", "encounters_cleaned_test")
