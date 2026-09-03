"""
The function-calling loop: sends the conversation + tool definitions to
the LLM, executes whichever tool(s) it calls, feeds results back, and
repeats until the LLM returns a final text answer.

Deliberately decoupled from any specific LLM SDK (OpenAI/Groq/etc). The
loop takes an `llm_call` function as a dependency rather than importing
an SDK directly — this is what makes it unit-testable without hitting a
real API (see tests/test_orchestrator.py), and keeps the actual
provider swappable later without touching this file.

Tool schemas aren't defined yet — that's the next step. For now this
loop works with whatever `tool_schemas` list and `tool_functions` dict
it's given; both can be empty during development.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

MAX_TOOL_CALL_ITERATIONS = 5


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMTurnResult:
    """
    What one call to the LLM produced. `raw_assistant_message` is the
    provider-specific message dict (e.g. OpenAI's assistant message
    format, including its own tool_calls representation) — the
    orchestrator appends it to the conversation as-is without needing
    to know its exact shape, so provider quirks stay inside the
    llm_client adapter, not in this loop.
    """

    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw_assistant_message: dict[str, Any] = field(default_factory=dict)


LLMCallFn = Callable[[list[dict], list[dict]], LLMTurnResult]


def make_tool_result_message(tool_call: ToolCall, result: Any) -> dict:
    return {
        "role": "tool",
        "tool_call_id": tool_call.id,
        "name": tool_call.name,
        "content": str(result),
    }


def run_conversation_turn(
    user_message: str,
    conversation_history: list[dict],
    tool_functions: dict[str, Callable[..., Any]],
    tool_schemas: list[dict],
    llm_call: LLMCallFn,
) -> str:
    """
    Runs one user turn through the function-calling loop until the LLM
    returns a final text answer (no more tool calls), or the iteration
    cap is hit.

    Returns the final answer text. Raises nothing on unknown tool names
    or tool execution errors — those are fed back to the LLM as the
    tool result so it can react, matching how real tool-calling APIs
    expect errors to be surfaced.
    """
    messages = conversation_history + [{"role": "user", "content": user_message}]

    for _ in range(MAX_TOOL_CALL_ITERATIONS):
        result = llm_call(messages, tool_schemas)

        if not result.tool_calls:
            return result.content or ""

        messages.append(result.raw_assistant_message)

        for tool_call in result.tool_calls:
            tool_result = _execute_tool(tool_call, tool_functions)
            messages.append(make_tool_result_message(tool_call, tool_result))

    return (
        "I wasn't able to finish answering that after several tool calls — "
        "could you try rephrasing your question?"
    )


def _execute_tool(tool_call: ToolCall, tool_functions: dict[str, Callable[..., Any]]) -> Any:
    fn = tool_functions.get(tool_call.name)
    if fn is None:
        return f"Error: unknown tool '{tool_call.name}'"
    try:
        return fn(**tool_call.arguments)
    except Exception as exc:  # noqa: BLE001 - deliberately broad: feed any tool error back to the LLM
        return f"Error executing tool '{tool_call.name}': {exc}"
