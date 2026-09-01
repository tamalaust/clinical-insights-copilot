"""
ETL entrypoint. Run with: python -m etl.run

Runs the full raw -> cleaned -> Postgres pipeline once. Safe to re-run
on demand (the load step replaces the cleaned table each time).
"""
from etl.clean import clean, load_raw
from etl.config import RAW_DATA_PATH
from etl.load import write_cleaned_table


def main() -> None:
    print(f"Loading raw data from {RAW_DATA_PATH}")
    raw_df = load_raw(RAW_DATA_PATH)
    print(f"Loaded {len(raw_df)} rows, {len(raw_df.columns)} columns")

    cleaned_df = clean(raw_df)
    outlier_cols = [c for c in cleaned_df.columns if c.endswith("_is_outlier")]
    for col in outlier_cols:
        flagged = cleaned_df[col].sum()
        print(f"  {col}: {flagged} rows flagged")

    write_cleaned_table(cleaned_df)
    print(f"Wrote {len(cleaned_df)} rows to the cleaned table")


if __name__ == "__main__":
    main()
