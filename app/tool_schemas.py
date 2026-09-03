"""
Tool schemas passed to the LLM's function-calling API so it knows what
tools exist and what arguments each one takes. Written in the standard
OpenAI-compatible function-calling format, which Groq's API also uses.

Kept as plain data (not tied to any tool's implementation) so this file
can be handed straight to the llm_client adapter's `tools=` parameter.
"""
from app.data_query import ALLOWED_GROUPBY_COLUMNS, ALLOWED_METRICS

_METRIC_ENUM = sorted(ALLOWED_METRICS)
_GROUPBY_ENUM = sorted(ALLOWED_GROUPBY_COLUMNS)

QUERY_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "query_data",
        "description": (
            "Query the cleaned hospital encounters dataset. Returns a "
            "value for the given metric, broken down by the given "
            "grouping column. Use this to answer factual questions "
            "about the data (e.g. 'what is the average time in "
            "hospital by admission type?')."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "metric": {
                    "type": "string",
                    "enum": _METRIC_ENUM,
                    "description": "The measure to compute.",
                },
                "groupby": {
                    "type": "string",
                    "enum": _GROUPBY_ENUM,
                    "description": "The column to break the metric down by.",
                },
            },
            "required": ["metric", "groupby"],
        },
    },
}

CHART_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "generate_chart",
        "description": (
            "Generate a chart of the given metric broken down by the "
            "given grouping column. Use this when the user asks to "
            "see, plot, chart, or visualize something (e.g. 'show "
            "readmission rate by age group')."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "metric": {
                    "type": "string",
                    "enum": _METRIC_ENUM,
                    "description": "The measure to chart.",
                },
                "groupby": {
                    "type": "string",
                    "enum": _GROUPBY_ENUM,
                    "description": "The column to break the metric down by.",
                },
                "chart_type": {
                    "type": "string",
                    "enum": ["bar", "pie", "line"],
                    "description": "The kind of chart to generate. Default to 'bar' unless the data or user request suggests otherwise.",
                },
            },
            "required": ["metric", "groupby", "chart_type"],
        },
    },
}

NARRATION_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "narrate_insight",
        "description": (
            "Explain a trend or pattern in the data using natural "
            "language. Use this when the user asks 'why' about a "
            "metric (e.g. 'why is the readmission rate higher for "
            "older patients?'), rather than just asking for the raw "
            "number."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "metric": {
                    "type": "string",
                    "enum": _METRIC_ENUM,
                    "description": "The measure to explain.",
                },
                "groupby": {
                    "type": "string",
                    "enum": _GROUPBY_ENUM,
                    "description": "The column the metric is broken down by.",
                },
                "question": {
                    "type": "string",
                    "description": "The user's original question, verbatim, for context.",
                },
            },
            "required": ["metric", "groupby", "question"],
        },
    },
}

ALL_TOOL_SCHEMAS = [QUERY_TOOL_SCHEMA, CHART_TOOL_SCHEMA, NARRATION_TOOL_SCHEMA]
