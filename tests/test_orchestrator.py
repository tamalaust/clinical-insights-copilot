from app.orchestrator import LLMTurnResult, ToolCall, run_conversation_turn


def test_no_tool_call_returns_immediately():
    """If the LLM answers directly with no tool calls, the loop should
    return that answer on the first call without touching any tools."""

    def fake_llm_call(messages, tool_schemas):
        return LLMTurnResult(content="Hello, I can help with that.")

    result = run_conversation_turn(
        user_message="hi",
        conversation_history=[],
        tool_functions={},
        tool_schemas=[],
        llm_call=fake_llm_call,
    )
    assert result == "Hello, I can help with that."


def test_single_tool_call_then_final_answer():
    """Simulates: LLM requests a tool call on turn 1, then answers using
    the tool result on turn 2."""
    call_count = {"n": 0}

    def fake_query_tool(**kwargs):
        assert kwargs == {"age_group": "[70-80)"}
        return {"readmission_rate_pct": 48.1}

    def fake_llm_call(messages, tool_schemas):
        call_count["n"] += 1
        if call_count["n"] == 1:
            tool_call = ToolCall(id="call_1", name="query_tool", arguments={"age_group": "[70-80)"})
            return LLMTurnResult(
                content=None,
                tool_calls=[tool_call],
                raw_assistant_message={"role": "assistant", "tool_calls": [{"id": "call_1"}]},
            )
        # second call: the tool result should now be in the message history
        tool_messages = [m for m in messages if m.get("role") == "tool"]
        assert len(tool_messages) == 1
        assert "48.1" in tool_messages[0]["content"]
        return LLMTurnResult(content="Readmission rate for that age group is 48.1%.")

    result = run_conversation_turn(
        user_message="what's the readmission rate for 70-80 year olds?",
        conversation_history=[],
        tool_functions={"query_tool": fake_query_tool},
        tool_schemas=[{"name": "query_tool"}],
        llm_call=fake_llm_call,
    )
    assert result == "Readmission rate for that age group is 48.1%."
    assert call_count["n"] == 2


def test_unknown_tool_name_is_fed_back_as_error_not_raised():
    """If the LLM asks for a tool that doesn't exist, the loop shouldn't
    crash — it should feed an error message back to the LLM."""
    call_count = {"n": 0}

    def fake_llm_call(messages, tool_schemas):
        call_count["n"] += 1
        if call_count["n"] == 1:
            tool_call = ToolCall(id="call_1", name="nonexistent_tool", arguments={})
            return LLMTurnResult(content=None, tool_calls=[tool_call], raw_assistant_message={"role": "assistant"})
        tool_messages = [m for m in messages if m.get("role") == "tool"]
        assert "unknown tool" in tool_messages[0]["content"].lower()
        return LLMTurnResult(content="Sorry, I couldn't do that.")

    result = run_conversation_turn(
        user_message="do something impossible",
        conversation_history=[],
        tool_functions={},
        tool_schemas=[],
        llm_call=fake_llm_call,
    )
    assert result == "Sorry, I couldn't do that."


def test_tool_execution_error_is_caught_and_fed_back():
    def broken_tool(**kwargs):
        raise ValueError("something went wrong")

    call_count = {"n": 0}

    def fake_llm_call(messages, tool_schemas):
        call_count["n"] += 1
        if call_count["n"] == 1:
            tool_call = ToolCall(id="call_1", name="broken_tool", arguments={})
            return LLMTurnResult(content=None, tool_calls=[tool_call], raw_assistant_message={"role": "assistant"})
        tool_messages = [m for m in messages if m.get("role") == "tool"]
        assert "something went wrong" in tool_messages[0]["content"]
        return LLMTurnResult(content="There was an error.")

    result = run_conversation_turn(
        user_message="trigger the bug",
        conversation_history=[],
        tool_functions={"broken_tool": broken_tool},
        tool_schemas=[],
        llm_call=fake_llm_call,
    )
    assert result == "There was an error."


def test_max_iterations_cap_prevents_infinite_loop():
    """If the LLM keeps requesting tool calls forever, the loop must
    still terminate rather than looping indefinitely."""

    def fake_llm_call(messages, tool_schemas):
        # always request another tool call, never a final answer
        tool_call = ToolCall(id="call_x", name="noop_tool", arguments={})
        return LLMTurnResult(content=None, tool_calls=[tool_call], raw_assistant_message={"role": "assistant"})

    result = run_conversation_turn(
        user_message="loop forever",
        conversation_history=[],
        tool_functions={"noop_tool": lambda **kw: "ok"},
        tool_schemas=[],
        llm_call=fake_llm_call,
    )
    assert "wasn't able to finish" in result
