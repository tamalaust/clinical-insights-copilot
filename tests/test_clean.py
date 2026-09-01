import pandas as pd

from etl.clean import clean, fix_dtypes, flag_outliers


def make_sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time_in_hospital": [1, 2, 3, 4, 5, 6, 7, 8, 9, 100],  # 100 is an outlier
            "num_lab_procedures": [10, 12, 11, 13, 12, 14, 11, 10, 13, 12],
            "race": ["Caucasian", "AfricanAmerican", "?", "Caucasian", "Asian",
                     "Caucasian", "?", "Hispanic", "Caucasian", "Other"],
        }
    )


def test_fix_dtypes_coerces_numeric_columns():
    df = pd.DataFrame({"time_in_hospital": ["1", "2", "not_a_number"]})
    result = fix_dtypes(df)
    assert pd.api.types.is_numeric_dtype(result["time_in_hospital"])
    assert result["time_in_hospital"].isna().sum() == 1


def test_flag_outliers_adds_boolean_column_without_dropping_rows():
    df = make_sample_df()
    result = flag_outliers(df)

    assert "time_in_hospital_is_outlier" in result.columns
    assert len(result) == len(df)  # no rows dropped
    assert result["time_in_hospital_is_outlier"].dtype == bool


def test_flag_outliers_flags_the_extreme_value():
    df = make_sample_df()
    result = flag_outliers(df)

    # the row with time_in_hospital == 100 should be flagged
    flagged_row = result[result["time_in_hospital"] == 100]
    assert flagged_row["time_in_hospital_is_outlier"].iloc[0] == True  # noqa: E712


def test_decode_id_columns_adds_description_columns():
    from etl.clean import decode_id_columns

    df = pd.DataFrame({"admission_type_id": [1, 2, 6]})
    result = decode_id_columns(df, mappings_path="data/raw/IDS_mapping.csv")

    assert "admission_type_id_description" in result.columns
    assert result.loc[result["admission_type_id"] == 1, "admission_type_id_description"].iloc[0] == "Emergency"


def test_clean_end_to_end_happy_path():
    df = make_sample_df()
    result = clean(df)

    # cleaning should never drop rows
    assert len(result) == len(df)
    # outlier columns should be present for configured numeric columns
    assert "time_in_hospital_is_outlier" in result.columns
    assert "num_lab_procedures_is_outlier" in result.columns
