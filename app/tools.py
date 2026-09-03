"""
The actual Python functions behind the three tool schemas in
tool_schemas.py. These are what go into the `tool_functions` dict
passed to app/orchestrator.py's run_conversation_turn().

query_data and generate_chart are plain functions — the orchestrator
calls them directly with whatever arguments the LLM provided.

narrate_insight is different: it needs to make its own LLM call to
generate the explanation text, but the orchestrator only ever calls
tool functions with the arguments the LLM supplied (metric, groupby,
question) — it has no way to also hand over an llm_call reference at
call time. So narrate_insight is built by a factory function,
make_narration_tool(llm_call), which captures llm_call in a closure.
This keeps the orchestrator itself completely unaware that one tool
happens to be LLM-backed.
"""
from __future__ import annotations

from typing import Any

from app.cache import with_exact_cache
from app.data_query import aggregate
from app.orchestrator import LLMCallFn
from app.data_access import read_cleaned_table
from app.semantic_cache import find_similar_narration, store_narration


def query_data(metric: str, groupby: str) -> list[dict[str, Any]]:
    """Returns the aggregated metric-by-group data as a list of dicts."""
    df = read_cleaned_table()
    result_df = aggregate(df, metric, groupby)
    return result_df.to_dict(orient="records")


def generate_chart(metric: str, groupby: str, chart_type: str) -> dict[str, Any]:
    """
    Returns a chart spec: the chart type plus the same aggregated data
    query_data would return. The UI layer is responsible for actually
    rendering this (e.g. handing it to Plotly) — this tool only
    produces the spec/data, per the project's original design.
    """
    df = read_cleaned_table()
    result_df = aggregate(df, metric, groupby)
    return {
        "chart_type": chart_type,
        "metric": metric,
        "groupby": groupby,
        "data": result_df.to_dict(orient="records"),
    }


def make_narration_tool(llm_call: LLMCallFn):
    """
    Returns a narrate_insight(metric, groupby, question) function with
    llm_call baked in via closure. Call this once at app startup with
    your real LLM adapter, and register the result in tool_functions
    under the name "narrate_insight" (matching NARRATION_TOOL_SCHEMA).

    Checks the semantic cache first (a similarly-worded question asked
    before for this metric/groupby skips the LLM call entirely); on a
    miss, calls the LLM and stores the result for next time.
    """

    def narrate_insight(metric: str, groupby: str, question: str) -> str:
        cached = find_similar_narration(metric, groupby, question)
        if cached is not None:
            return cached

        df = read_cleaned_table()
        result_df = aggregate(df, metric, groupby)
        data_summary = result_df.to_dict(orient="records")

        prompt = (
            f"A user asked: \"{question}\"\n\n"
            f"Here is the underlying data ({metric} by {groupby}):\n"
            f"{data_summary}\n\n"
            f"In 2-3 sentences, explain the pattern in this data in "
            f"plain language a hospital administrator could understand. "
            f"Base your explanation only on the numbers shown above — "
            f"do not add outside medical facts or general knowledge not "
            f"present in this data. Do not give clinical or diagnostic "
            f"advice — describe the pattern in the aggregate data only."
        )
        narration_messages = [{"role": "user", "content": prompt}]
        result = llm_call(narration_messages, [])
        narration = result.content or "I wasn't able to generate an explanation for that."

        store_narration(metric, groupby, question, narration)
        return narration

    return narrate_insight


def make_tool_functions(llm_call: LLMCallFn) -> dict[str, Any]:
    """
    Convenience factory: builds the full tool_functions dict expected
    by run_conversation_turn(), with the narration tool wired to the
    given LLM adapter, and query_data/generate_chart wrapped with
    exact-match Redis caching.
    """
    return {
        "query_data": with_exact_cache("query_data", query_data),
        "generate_chart": with_exact_cache("generate_chart", generate_chart),
        "narrate_insight": make_narration_tool(llm_call),
    }