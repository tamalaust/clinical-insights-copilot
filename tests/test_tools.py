from unittest.mock import patch

import pandas as pd

from app.orchestrator import LLMTurnResult
from app.tools import generate_chart, make_narration_tool, query_data


def make_sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "gender": ["Female", "Female", "Male", "Male"],
            "readmitted": ["<30", "NO", "NO", ">30"],
            "time_in_hospital": [3, 5, 1, 2],
            "num_medications": [10, 20, 5, 7],
            "num_lab_procedures": [40, 50, 30, 20],
            "age": ["[70-80)", "[70-80)", "[20-30)", "[20-30)"],
            "admission_type_id_description": ["Emergency"] * 4,
        }
    )


@patch("app.tools.read_cleaned_table")
def test_query_data_returns_list_of_dicts(mock_read):
    mock_read.return_value = make_sample_df()
    result = query_data(metric="readmission_rate", groupby="gender")

    assert isinstance(result, list)
    assert all(isinstance(row, dict) for row in result)
    female_row = next(r for r in result if r["gender"] == "Female")
    assert female_row["readmission_rate"] == 50.0  # 1 of 2 readmitted


@patch("app.tools.read_cleaned_table")
def test_generate_chart_includes_chart_type_and_data(mock_read):
    mock_read.return_value = make_sample_df()
    result = generate_chart(metric="avg_num_medications", groupby="gender", chart_type="bar")

    assert result["chart_type"] == "bar"
    assert result["metric"] == "avg_num_medications"
    assert result["groupby"] == "gender"
    assert isinstance(result["data"], list)
    assert len(result["data"]) == 2  # Female, Male


@patch("app.tools.read_cleaned_table")
def test_narrate_insight_calls_llm_with_data_summary_and_returns_content(mock_read):
    mock_read.return_value = make_sample_df()

    captured_prompt = {}

    def fake_llm_call(messages, tool_schemas):
        captured_prompt["text"] = messages[0]["content"]
        assert tool_schemas == []  # narration doesn't offer further tool calls
        return LLMTurnResult(content="Female patients had a higher readmission rate in this sample.")

    narrate_insight = make_narration_tool(fake_llm_call)
    result = narrate_insight(metric="readmission_rate", groupby="gender", question="why the difference?")

    assert result == "Female patients had a higher readmission rate in this sample."
    # the prompt sent to the LLM should include the user's question and the data
    assert "why the difference?" in captured_prompt["text"]
    assert "readmission_rate" in captured_prompt["text"]
