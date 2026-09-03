"""
Generic aggregation logic for the agent's tools. Unlike ui/charts.py
(which has 4 fixed charts for the static dashboard), the chat agent
needs to answer arbitrary "metric by groupby" questions — e.g. "average
medications by gender" or "readmission rate by admission type" — so
this supports any valid combination instead of one function per chart.

Kept as pure functions (DataFrame in, DataFrame out) for the same
testability reasons as etl/clean.py and ui/charts.py.
"""
from __future__ import annotations

import pandas as pd

# Columns the agent is allowed to group by. Deliberately an allowlist,
# not "any column in the DataFrame" — keeps the agent from grouping by
# something meaningless (e.g. encounter_id) or a raw ID code instead of
# its decoded description.
ALLOWED_GROUPBY_COLUMNS = {
    "age",
    "gender",
    "race",
    "admission_type_id_description",
    "discharge_disposition_id_description",
    "admission_source_id_description",
}

# Metric name -> (source column, aggregation). "readmission_rate" is
# special-cased since it's derived (readmitted != 'NO'), not a direct
# column mean.
ALLOWED_METRICS = {
    "readmission_rate",
    "avg_time_in_hospital",
    "avg_num_medications",
    "avg_num_lab_procedures",
    "encounter_count",
}


def aggregate(df: pd.DataFrame, metric: str, groupby: str) -> pd.DataFrame:
    """
    Returns a DataFrame with one row per group: [groupby, metric_value].
    Raises ValueError for an unsupported metric/groupby — this is
    caught by the orchestrator's tool-error handling and fed back to
    the LLM as a normal tool error, not a crash.
    """
    if groupby not in ALLOWED_GROUPBY_COLUMNS:
        raise ValueError(
            f"Unsupported groupby column '{groupby}'. "
            f"Allowed: {sorted(ALLOWED_GROUPBY_COLUMNS)}"
        )
    if metric not in ALLOWED_METRICS:
        raise ValueError(
            f"Unsupported metric '{metric}'. Allowed: {sorted(ALLOWED_METRICS)}"
        )

    grouped = df.groupby(groupby, observed=True)

    if metric == "readmission_rate":
        result = grouped.apply(
            lambda g: (g["readmitted"] != "NO").mean() * 100,
            include_groups=False,
        )
    elif metric == "avg_time_in_hospital":
        result = grouped["time_in_hospital"].mean()
    elif metric == "avg_num_medications":
        result = grouped["num_medications"].mean()
    elif metric == "avg_num_lab_procedures":
        result = grouped["num_lab_procedures"].mean()
    elif metric == "encounter_count":
        result = grouped.size()

    return (
        result.dropna()
        .reset_index()
        .rename(columns={0: metric, groupby: groupby})
        .set_axis([groupby, metric], axis=1)
    )
