"""Launch-verification repros for the 2026-10-18 LangGraph dogfood findings.

Each test reproduces the *exact* scenario from the Linear issue, not a unit of
the fix, on a fake chat model (no network, no LLM keys):

- MHQ-1402: the documented zero-config integration (``morse_ai.init()`` only)
  on an async graph served the way FastAPI serves it — several requests
  running ``ainvoke`` / ``astream`` concurrently on one event loop, no
  ``morse_ai.run()``. Each request must get its own trace with every span and
  its tokens.
- MHQ-1403: the published snippet (zero-config *plus* an explicit
  ``MorseCallbackHandler`` in ``config``) on a tool-calling ReAct graph with
  several model calls. Exactly one ``llm`` span per real model call.
- MHQ-1405: a real supervisor handoff, ``Command(graph=Command.PARENT)``
  raised from inside a subgraph, must leave every span and the trace
  ``completed`` with no ``exception_type``.
"""

# ruff: noqa: E501
from __future__ import annotations

import asyncio
import warnings
from typing import Any, TypedDict

import pytest

pytest.importorskip("langgraph")
pytest.importorskip("langchain_core")

from langchain_core.language_models.chat_models import BaseChatModel  # noqa: E402
from langchain_core.messages import AIMessage, ToolMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402
from langchain_core.tools import tool  # noqa: E402
from langgraph.graph import END, START, StateGraph  # noqa: E402
from langgraph.types import Command  # noqa: E402

from morse_ai import tracing  # noqa: E402
from morse_ai.adapters import langgraph as lga  # noqa: E402
from morse_ai.adapters.langchain import MorseCallbackHandler  # noqa: E402


class _ToolCallingFake(BaseChatModel):
    """Calls ``add`` once, then answers. Usage on every call (10 in / 5 out)."""

    @property
    def _llm_type(self) -> str:
        return "fake-tool-calling"

    def bind_tools(self, tools, **kwargs):  # noqa: ANN001, ANN003
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):  # noqa: ANN001
        already_called = any(isinstance(m, ToolMessage) for m in messages)
        usage = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
        meta = {"model_name": "gpt-4o-mini"}
        if already_called:
            msg = AIMessage(content="42", usage_metadata=usage, response_metadata=meta)
        else:
            msg = AIMessage(
                content="",
                tool_calls=[{"name": "add", "args": {"a": 40, "b": 2}, "id": "call_1"}],
                usage_metadata=usage,
                response_metadata=meta,
            )
        return ChatResult(generations=[ChatGeneration(message=msg)])


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


def _react_graph():
    from langgraph.prebuilt import create_react_agent

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return create_react_agent(_ToolCallingFake(), [add], name="calc_agent")


@pytest.fixture(autouse=True)
def _zero_config():
    """Reproduce ``morse_ai.init()``'s auto-install, nothing else."""
    token = tracing._active_trace.set(None)
    lga.uninstall()
    tracing._last_trace = None
    lga.install()
    try:
        yield
    finally:
        lga.uninstall()
        tracing._active_trace.reset(token)


def _capture_traces(monkeypatch) -> list[dict[str, Any]]:  # noqa: ANN001
    """Collect every finished trace payload (concurrent requests overwrite
    ``_last_trace``)."""
    seen: list[dict[str, Any]] = []
    real = tracing._capture_last_trace

    def spy(entry):  # noqa: ANN001
        out = real(entry)
        seen.append(out)
        return out

    monkeypatch.setattr(tracing, "_capture_last_trace", spy)
    return seen


def _llm_spans(spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [s for s in spans if s["type"] == "llm"]


# ---------------------------------------------------------------------------
# MHQ-1402
# ---------------------------------------------------------------------------


@pytest.mark.req("SDK-09")
def test_mhq1402_documented_zero_config_async_graph_records_full_trace():
    """``asyncio.run(graph.ainvoke(...))`` with init() only (the issue's repro)."""
    asyncio.run(_react_graph().ainvoke({"messages": [("user", "40+2?")]}))
    trace = tracing._get_last_trace()
    assert trace is not None
    spans = trace["spans"]
    llm = _llm_spans(spans)
    # 2 model calls (tool call, then answer) + 1 tool call; before the fix: 2 spans, 0 tokens.
    assert len(llm) == 2
    assert [s for s in spans if s["type"] == "tool"]
    assert sum(s.get("input_tokens") or 0 for s in llm) == 20
    assert all(s.get("model") for s in llm)
    assert len({s["trace_id"] for s in spans}) == 1


@pytest.mark.req("SDK-09")
def test_mhq1402_concurrent_fastapi_style_requests_get_separate_full_traces(
    monkeypatch,
):
    """Three requests served concurrently on one loop (oss1 agent-service shape)."""
    seen = _capture_traces(monkeypatch)
    graph = _react_graph()

    async def handle_invoke(i: int):
        return await graph.ainvoke({"messages": [("user", f"q{i}")]})

    async def handle_stream(i: int):
        async for _ in graph.astream({"messages": [("user", f"s{i}")]}):
            pass

    async def server():
        await asyncio.gather(handle_invoke(1), handle_invoke(2), handle_stream(3))

    asyncio.run(server())

    assert len(seen) == 3, [len(t["spans"]) for t in seen]
    for payload in seen:
        llm = _llm_spans(payload["spans"])
        assert len(llm) == 2, "each request: two model calls, one span each"
        assert sum(s.get("input_tokens") or 0 for s in llm) == 20
        assert len({s["trace_id"] for s in payload["spans"]}) == 1
    assert len({p["spans"][0]["trace_id"] for p in seen}) == 3


# ---------------------------------------------------------------------------
# MHQ-1403
# ---------------------------------------------------------------------------


@pytest.mark.req("SDK-09")
@pytest.mark.parametrize("runner", ["sync", "async"])
def test_mhq1403_published_snippet_init_plus_handler_one_llm_span_per_call(runner):
    """init() auto-install + ``config={"callbacks": [MorseCallbackHandler()]}``."""
    cfg = {"callbacks": [MorseCallbackHandler(agent_name="calc")]}
    graph = _react_graph()
    inp = {"messages": [("user", "40+2?")]}
    if runner == "sync":
        graph.invoke(inp, config=cfg)
    else:
        asyncio.run(graph.ainvoke(inp, config=cfg))
    trace = tracing._get_last_trace()
    assert trace is not None
    llm = _llm_spans(trace["spans"])
    assert len(llm) == 2  # before the fix: 6 (three per call)
    assert sum(s.get("input_tokens") or 0 for s in llm) == 20


@pytest.mark.req("SDK-09")
def test_mhq1403_zero_config_alone_has_no_duplicate_sibling_llm_span():
    _react_graph().invoke({"messages": [("user", "40+2?")]})
    trace = tracing._get_last_trace()
    assert trace is not None
    llm = _llm_spans(trace["spans"])
    assert len(llm) == 2
    assert sum(s.get("input_tokens") or 0 for s in llm) == 20


# ---------------------------------------------------------------------------
# MHQ-1405
# ---------------------------------------------------------------------------


class _S(TypedDict, total=False):
    answer: str
    route: str


def _supervisor_graph():
    """Parent graph: ``worker`` is a subgraph whose node hands off to the
    parent's ``finish`` node with ``Command(graph=Command.PARENT)``; langgraph
    implements that by raising ``ParentCommand`` through the callbacks."""

    def handoff(state: _S):  # noqa: ANN202
        return Command(graph=Command.PARENT, goto="finish", update={"route": "handoff"})

    sub = StateGraph(_S)
    sub.add_node("handoff", handoff)
    sub.add_edge(START, "handoff")
    worker = sub.compile()

    def finish(state: _S):  # noqa: ANN202
        return {"answer": "42"}

    parent = StateGraph(_S)
    parent.add_node("worker", worker)
    parent.add_node("finish", finish)
    parent.add_edge(START, "worker")
    parent.add_edge("finish", END)
    return parent.compile(name="supervisor")


@pytest.mark.req("SDK-09")
@pytest.mark.parametrize("runner", ["sync", "async"])
def test_mhq1405_supervisor_handoff_parent_command_is_not_a_failure(runner):
    graph = _supervisor_graph()
    out = graph.invoke({}) if runner == "sync" else asyncio.run(graph.ainvoke({}))
    assert out["answer"] == "42"
    trace = tracing._get_last_trace()
    assert trace is not None
    assert trace["trace"]["status"] == "completed"
    assert not trace["trace"].get("error_message")
    for s in trace["spans"]:
        assert s["status"] == "completed", (s["name"], s.get("exception_type"))
        assert not s.get("exception_type")
