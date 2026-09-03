"""
FastAPI backend. Exposes the agentic chat loop as a single POST /chat
endpoint. This is the piece that finally wires together everything
built so far: app/orchestrator.py (the loop), app/tools.py (the 3
tools), app/tool_schemas.py (what the LLM is told about them), and
app/llm_client.py (the actual Ollama call).

Run with: uvicorn app.main:app --reload
"""
from fastapi import FastAPI
from pydantic import BaseModel

from app.cache import reset_cache_hit_flag, was_served_from_cache
from app.llm_client import make_ollama_llm_call
from app.orchestrator import run_conversation_turn
from app.tool_schemas import ALL_TOOL_SCHEMAS
from app.tools import make_tool_functions

app = FastAPI(title="Clinical Insights Copilot API")

# Built once at startup, not per-request — the Ollama client and the
# narration tool's closure over it are reused across requests.
_llm_call = make_ollama_llm_call()
_tool_functions = make_tool_functions(_llm_call)

# Constrains the LLM to only answer using the tools/dataset, not its
# own general pretrained knowledge. Without this, the model will
# sometimes skip calling a tool and answer directly from training
# data when a question doesn't map cleanly onto a tool call — which
# looks like it "knows" something about the data when it's really
# just generating plausible-sounding, ungrounded text.
SYSTEM_PROMPT = (
    "You are Clinical Insights Copilot, an assistant for exploring a "
    "specific aggregate, non-identifiable hospital encounters dataset. "
    "Only answer using the tools provided — they query the actual "
    "dataset. Do not use your own general medical or clinical "
    "knowledge to answer questions; you only know what the tools "
    "return. If a question can't be answered with the available "
    "tools, say so plainly instead of guessing. Do not give clinical, "
    "diagnostic, or medical advice — describe only patterns present "
    "in the aggregate data."
)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[ChatMessage] = []


class ChatResponse(BaseModel):
    answer: str
    from_cache: bool = False


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    reset_cache_hit_flag()

    conversation_history = [{"role": "system", "content": SYSTEM_PROMPT}] + [
        m.model_dump() for m in request.history
    ]
    answer = run_conversation_turn(
        user_message=request.message,
        conversation_history=conversation_history,
        tool_functions=_tool_functions,
        tool_schemas=ALL_TOOL_SCHEMAS,
        llm_call=_llm_call,
    )
    return ChatResponse(answer=answer, from_cache=was_served_from_cache())