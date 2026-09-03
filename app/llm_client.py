"""
Implements the llm_call interface (see app/orchestrator.py's LLMCallFn)
against a local Ollama server, using Ollama's OpenAI-compatible
/v1/chat/completions endpoint — so this uses the standard `openai`
Python SDK, just pointed at localhost instead of OpenAI's API.

This is the one place in the project that knows about Ollama/OpenAI-
specific response shapes (tool_calls, function.arguments as a JSON
string, etc). Everything else — the orchestrator, the tools — only
ever sees the provider-agnostic LLMTurnResult/ToolCall types.
"""
from __future__ import annotations

import json

from openai import OpenAI

from app.config import OLLAMA_BASE_URL, OLLAMA_MODEL
from app.orchestrator import LLMTurnResult, ToolCall


def make_ollama_llm_call(model: str | None = None, client: OpenAI | None = None):
    """
    Returns an llm_call function (matching LLMCallFn) bound to the given
    model and client. `client` is injectable so this is testable with a
    fake client instead of a real Ollama server — see
    tests/test_llm_client.py.
    """
    resolved_model = model or OLLAMA_MODEL
    resolved_client = client or OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")

    def llm_call(messages: list[dict], tool_schemas: list[dict]) -> LLMTurnResult:
        request_kwargs = {"model": resolved_model, "messages": messages}
        if tool_schemas:
            request_kwargs["tools"] = tool_schemas

        response = resolved_client.chat.completions.create(**request_kwargs)
        message = response.choices[0].message

        tool_calls = []
        raw_tool_calls = []
        if message.tool_calls:
            for tc in message.tool_calls:
                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=json.loads(tc.function.arguments or "{}"),
                    )
                )
                raw_tool_calls.append(
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                )

        raw_assistant_message = {
            "role": "assistant",
            "content": message.content,
        }
        if raw_tool_calls:
            raw_assistant_message["tool_calls"] = raw_tool_calls

        return LLMTurnResult(
            content=message.content,
            tool_calls=tool_calls,
            raw_assistant_message=raw_assistant_message,
        )

    return llm_call
