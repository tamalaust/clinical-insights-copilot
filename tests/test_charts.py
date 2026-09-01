import pandas as pd

from ui.charts import (
    admission_type_breakdown,
    avg_medications_by_age,
    readmission_rate_by_age,
    time_in_hospital_distribution,
)


def make_sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": ["[70-80)", "[70-80)", "[20-30)", "[20-30)", "[20-30)"],
            "readmitted": ["<30", "NO", "NO", "NO", ">30"],
            "time_in_hospital": [3, 5, 1, 1, 2],
            "num_medications": [10, 20, 5, 7, 6],
            "admission_type_id_description": ["Emergency", "Emergency", "Urgent", "Elective", "Emergency"],
        }
    )


def test_readmission_rate_by_age_computes_correct_percentage():
    df = make_sample_df()
    result = readmission_rate_by_age(df)

    # [70-80): 1 of 2 readmitted -> 50%
    row_70_80 = result[result["age"] == "[70-80)"]
    assert row_70_80["readmission_rate_pct"].iloc[0] == 50.0

    # [20-30): 1 of 3 readmitted -> ~33.3%
    row_20_30 = result[result["age"] == "[20-30)"]
    assert round(row_20_30["readmission_rate_pct"].iloc[0], 1) == 33.3


def test_readmission_rate_by_age_only_returns_present_buckets():
    df = make_sample_df()
    result = readmission_rate_by_age(df)
    # only 2 age buckets are present in the sample data
    assert len(result) == 2


def test_time_in_hospital_distribution_counts_correctly():
    df = make_sample_df()
    result = time_in_hospital_distribution(df)
    row_1 = result[result["time_in_hospital"] == 1]
    assert row_1["encounter_count"].iloc[0] == 2


def test_admission_type_breakdown_counts_correctly():
    df = make_sample_df()
    result = admission_type_breakdown(df)
    row = result[result["admission_type_id_description"] == "Emergency"]
    assert row["encounter_count"].iloc[0] == 3


def test_avg_medications_by_age_computes_mean():
    df = make_sample_df()
    result = avg_medications_by_age(df)
    row_70_80 = result[result["age"] == "[70-80)"]
    assert row_70_80["avg_num_medications"].iloc[0] == 15.0  # (10+20)/2
