import json

from app.llm_client import make_ollama_llm_call


class FakeFunction:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments  # JSON string, matching the real SDK


class FakeToolCall:
    def __init__(self, id, name, arguments):
        self.id = id
        self.function = FakeFunction(name, arguments)


class FakeMessage:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class FakeChoice:
    def __init__(self, message):
        self.message = message


class FakeResponse:
    def __init__(self, message):
        self.choices = [FakeChoice(message)]


class FakeCompletions:
    def __init__(self, response, captured_kwargs):
        self._response = response
        self._captured_kwargs = captured_kwargs

    def create(self, **kwargs):
        self._captured_kwargs.update(kwargs)
        return self._response


class FakeChat:
    def __init__(self, completions):
        self.completions = completions


class FakeClient:
    def __init__(self, response):
        self.captured_kwargs = {}
        self.chat = FakeChat(FakeCompletions(response, self.captured_kwargs))


def test_plain_text_response_no_tool_calls():
    fake_message = FakeMessage(content="Hello there!", tool_calls=None)
    fake_client = FakeClient(FakeResponse(fake_message))

    llm_call = make_ollama_llm_call(model="llama3.2", client=fake_client)
    result = llm_call([{"role": "user", "content": "hi"}], [])

    assert result.content == "Hello there!"
    assert result.tool_calls == []
    assert result.raw_assistant_message == {"role": "assistant", "content": "Hello there!"}


def test_tool_schemas_omitted_from_request_when_empty():
    fake_message = FakeMessage(content="ok")
    fake_client = FakeClient(FakeResponse(fake_message))

    llm_call = make_ollama_llm_call(model="llama3.2", client=fake_client)
    llm_call([{"role": "user", "content": "hi"}], [])

    assert "tools" not in fake_client.captured_kwargs


def test_tool_schemas_included_in_request_when_present():
    fake_message = FakeMessage(content="ok")
    fake_client = FakeClient(FakeResponse(fake_message))
    schemas = [{"type": "function", "function": {"name": "query_data"}}]

    llm_call = make_ollama_llm_call(model="llama3.2", client=fake_client)
    llm_call([{"role": "user", "content": "hi"}], schemas)

    assert fake_client.captured_kwargs["tools"] == schemas


def test_tool_call_response_parsed_into_toolcall_objects():
    fake_tool_call = FakeToolCall(
        id="call_abc123",
        name="query_data",
        arguments=json.dumps({"metric": "readmission_rate", "groupby": "gender"}),
    )
    fake_message = FakeMessage(content=None, tool_calls=[fake_tool_call])
    fake_client = FakeClient(FakeResponse(fake_message))

    llm_call = make_ollama_llm_call(model="llama3.2", client=fake_client)
    result = llm_call([{"role": "user", "content": "show me readmission by gender"}], [])

    assert len(result.tool_calls) == 1
    tc = result.tool_calls[0]
    assert tc.id == "call_abc123"
    assert tc.name == "query_data"
    assert tc.arguments == {"metric": "readmission_rate", "groupby": "gender"}


def test_tool_call_raw_assistant_message_has_openai_shape_for_replay():
    """The raw_assistant_message must be appendable back into the
    conversation history in a shape the API will accept on the next
    turn — this checks that structure, not just the parsed ToolCall."""
    fake_tool_call = FakeToolCall(id="call_1", name="generate_chart", arguments="{}")
    fake_message = FakeMessage(content=None, tool_calls=[fake_tool_call])
    fake_client = FakeClient(FakeResponse(fake_message))

    llm_call = make_ollama_llm_call(model="llama3.2", client=fake_client)
    result = llm_call([{"role": "user", "content": "chart it"}], [])

    assert result.raw_assistant_message["role"] == "assistant"
    assert result.raw_assistant_message["tool_calls"][0]["id"] == "call_1"
    assert result.raw_assistant_message["tool_calls"][0]["function"]["name"] == "generate_chart"
