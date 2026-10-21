"""LangGraph adapter: async zero-config tracing, LLM de-duplication, control flow.

MHQ-1402: ``asyncio.run(graph.ainvoke(...))`` without ``morse_ai.run()`` stored
only 2 spans. LangChain runs a *sync* handler's callbacks from an async path in
an executor with a *copied* context, so the ContextVar the auto-trace sets in
``on_chain_start`` was invisible to every later callback. The handler now
declares ``run_inline = True`` so LangChain calls it on the event loop thread.

MHQ-1403: one real LLM call produced 2-3 identical ``llm`` spans when the
handler was attached more than once (explicit + auto-installed + inherited).
Handlers now claim each LangChain ``run_id``; the first claimant records it.

MHQ-1405: ``langgraph.errors.ParentCommand`` / ``GraphInterrupt`` and friends
are control flow, not failures. They must not mark spans or traces failed.

These tests build real (tiny) LangGraph graphs on a fake chat model, so they
need ``langgraph`` + ``langchain_core`` and skip cleanly without them.
"""

# ruff: noqa: E501
from __future__ import annotations

import asyncio
import uuid
from collections import Counter
from typing import Any, TypedDict

import pytest

pytest.importorskip("langgraph")
pytest.importorskip("langchain_core")

from langchain_core.language_models.chat_models import BaseChatModel  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402
from langchain_core.tools import tool  # noqa: E402
from langgraph.graph import END, START, StateGraph  # noqa: E402

from morse_ai import tracing  # noqa: E402
from morse_ai.adapters import langgraph as lga  # noqa: E402
from morse_ai.adapters.langchain import MorseCallbackHandler  # noqa: E402
from morse_ai.adapters.langgraph import LangGraphCallbackHandler  # noqa: E402


class _FakeChat(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "fake"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):  # noqa: ANN001
        msg = AIMessage(
            content="hi",
            usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            response_metadata={"model_name": "gpt-4o-mini"},
        )
        return ChatResult(generations=[ChatGeneration(message=msg)])


@tool
def _add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


class _State(TypedDict):
    x: str


_llm = _FakeChat()


def _build(async_nodes: bool):
    def llm_sync(s):  # noqa: ANN001
        _llm.invoke("q")
        return {"x": "a"}

    async def llm_async(s):  # noqa: ANN001
        await _llm.ainvoke("q")
        return {"x": "a"}

    def tool_sync(s):  # noqa: ANN001
        _add.invoke({"a": 1, "b": 2})
        return {"x": "b"}

    async def tool_async(s):  # noqa: ANN001
        await _add.ainvoke({"a": 1, "b": 2})
        return {"x": "b"}

    g = StateGraph(_State)
    g.add_node("llm_node", llm_async if async_nodes else llm_sync)
    g.add_node("tool_node", tool_async if async_nodes else tool_sync)
    g.add_edge(START, "llm_node")
    g.add_edge("llm_node", "tool_node")
    g.add_edge("tool_node", END)
    return g.compile()


@pytest.fixture(autouse=True)
def _clean():
    token = tracing._active_trace.set(None)
    lga.uninstall()
    tracing._last_trace = None
    try:
        yield
    finally:
        lga.uninstall()
        tracing._active_trace.reset(token)


def _spans() -> list[dict[str, Any]]:
    trace = tracing._get_last_trace()
    assert trace is not None, "no trace was emitted"
    return trace["spans"]


def _shape(spans: list[dict[str, Any]]) -> tuple[Counter, int]:
    return (
        Counter((s["type"], s["name"]) for s in spans if s["name"] != "chain"),
        sum(s.get("input_tokens") or 0 for s in spans),
    )


# ---------------------------------------------------------------------------
# MHQ-1402 — async without run()
# ---------------------------------------------------------------------------


@pytest.mark.req("SDK-09")
def test_async_graph_without_run_matches_sync_span_set():
    lga.install(agent_name="t")
    _build(False).invoke({"x": ""})
    sync_shape = _shape(_spans())
    tracing._last_trace = None

    asyncio.run(_build(True).ainvoke({"x": ""}))
    async_spans = _spans()

    assert _shape(async_spans) == sync_shape
    assert {s["type"] for s in async_spans} >= {"agent", "llm", "tool"}
    assert sum(s.get("input_tokens") or 0 for s in async_spans) == 10
    assert len({s["trace_id"] for s in async_spans}) == 1


@pytest.mark.req("SDK-09")
def test_handler_runs_inline_so_contextvars_reach_child_callbacks():
    assert LangGraphCallbackHandler.run_inline is True


# ---------------------------------------------------------------------------
# MHQ-1403 — one llm span per real call
# ---------------------------------------------------------------------------


def _llm_spans() -> list[dict[str, Any]]:
    return [s for s in _spans() if s["type"] == "llm"]


@pytest.mark.req("SDK-09")
@pytest.mark.parametrize("runner", ["sync", "async"])
def test_explicit_langgraph_handler_plus_auto_install_records_one_llm_span(runner):
    lga.install(agent_name="t")
    handler = LangGraphCallbackHandler(agent_name="t")
    cfg = {"callbacks": [handler]}
    if runner == "sync":
        _build(False).invoke({"x": ""}, config=cfg)
    else:
        asyncio.run(_build(True).ainvoke({"x": ""}, config=cfg))
    llm = _llm_spans()
    assert len(llm) == 1
    assert llm[0]["input_tokens"] == 10


@pytest.mark.req("SDK-09")
@pytest.mark.parametrize("runner", ["sync", "async"])
def test_explicit_base_handler_plus_auto_install_records_one_llm_span(runner):
    lga.install(agent_name="t")
    cfg = {"callbacks": [MorseCallbackHandler(agent_name="t")]}
    if runner == "sync":
        _build(False).invoke({"x": ""}, config=cfg)
    else:
        asyncio.run(_build(True).ainvoke({"x": ""}, config=cfg))
    llm = _llm_spans()
    assert len(llm) == 1
    assert llm[0]["input_tokens"] == 10


@pytest.mark.req("SDK-09")
def test_two_langgraph_handlers_record_one_llm_span():
    cfg = {
        "callbacks": [
            LangGraphCallbackHandler(agent_name="a"),
            LangGraphCallbackHandler(agent_name="a"),
        ]
    }
    with tracing.run("a"):
        _build(False).invoke({"x": ""}, config=cfg)
    assert len(_llm_spans()) == 1


@pytest.mark.req("SDK-09")
def test_same_run_id_started_twice_on_one_handler_opens_one_span():
    handler = LangGraphCallbackHandler(agent_name="a")
    rid = uuid.uuid4()
    with tracing.run("a"):
        for _ in range(3):
            handler.on_llm_start({"id": ["x", "FakeLLM"]}, ["p"], run_id=rid)
        handler.on_llm_end(_result(), run_id=rid)
        handler.on_llm_end(_result(), run_id=rid)
    assert len(_llm_spans()) == 1


@pytest.mark.req("SDK-09")
def test_same_run_id_on_two_handlers_counts_tokens_once():
    h1 = LangGraphCallbackHandler(agent_name="a")
    h2 = LangGraphCallbackHandler(agent_name="a")
    rid = uuid.uuid4()
    with tracing.run("a"):
        for h in (h1, h2):
            h.on_llm_start({"id": ["x", "FakeLLM"]}, ["p"], run_id=rid)
        for h in (h1, h2):
            h.on_llm_end(_result(), run_id=rid)
    llm = _llm_spans()
    assert len(llm) == 1
    assert llm[0]["input_tokens"] == 10


@pytest.mark.req("SDK-09")
def test_distinct_run_ids_are_not_deduplicated():
    handler = LangGraphCallbackHandler(agent_name="a")
    with tracing.run("a"):
        for _ in range(2):
            rid = uuid.uuid4()
            handler.on_llm_start({"id": ["x", "FakeLLM"]}, ["p"], run_id=rid)
            handler.on_llm_end(_result(), run_id=rid)
    assert len(_llm_spans()) == 2


def _result():
    msg = AIMessage(
        content="hi",
        usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        response_metadata={"model_name": "gpt-4o-mini"},
    )
    from langchain_core.outputs import LLMResult

    return LLMResult(generations=[[ChatGeneration(message=msg)]])


# ---------------------------------------------------------------------------
# MHQ-1405 — control-flow exceptions are not failures
# ---------------------------------------------------------------------------


def _control_flow_exceptions() -> list[BaseException]:
    from langgraph import errors

    out: list[BaseException] = []
    for name in ("GraphBubbleUp", "GraphInterrupt", "NodeInterrupt", "ParentCommand"):
        cls = getattr(errors, name, None)
        if cls is None:
            continue
        try:
            if name == "ParentCommand":
                out.append(cls(object()))
            elif name == "NodeInterrupt":
                out.append(cls("pause"))
            else:
                out.append(cls())
        except Exception:  # noqa: BLE001 - signature differs across versions
            continue
    assert out, "no langgraph control-flow exception could be built"
    return out


@pytest.mark.req("SDK-09")
@pytest.mark.parametrize(
    "exc", _control_flow_exceptions(), ids=lambda e: type(e).__name__
)
def test_control_flow_exception_does_not_fail_chain_span_or_auto_trace(exc):
    handler = LangGraphCallbackHandler(agent_name="a")
    root, child = uuid.uuid4(), uuid.uuid4()
    ser = {"id": ["langgraph", "graph", "CompiledStateGraph"]}
    handler.on_chain_start(ser, {"q": 1}, run_id=root)
    handler.on_chain_start(ser, {"q": 1}, run_id=child, parent_run_id=root)
    handler.on_chain_error(exc, run_id=child, parent_run_id=root)
    handler.on_chain_end({"a": 1}, run_id=root)

    trace = tracing._get_last_trace()
    assert trace is not None
    assert trace["trace"]["status"] == "completed"
    assert not trace["trace"].get("error_message")
    for s in trace["spans"]:
        assert s["status"] == "completed"
        assert not s.get("exception_type")
        assert not s.get("error_message")


@pytest.mark.req("SDK-09")
def test_control_flow_exception_on_root_chain_error_keeps_trace_completed():
    from langgraph.errors import ParentCommand

    handler = LangGraphCallbackHandler(agent_name="a")
    root = uuid.uuid4()
    handler.on_chain_start({"id": ["langgraph", "g", "G"]}, {}, run_id=root)
    handler.on_chain_error(ParentCommand(object()), run_id=root)
    trace = tracing._get_last_trace()
    assert trace is not None
    assert trace["trace"]["status"] == "completed"
    assert all(not s.get("exception_type") for s in trace["spans"])


@pytest.mark.req("SDK-09")
def test_real_errors_still_fail():
    handler = LangGraphCallbackHandler(agent_name="a")
    root = uuid.uuid4()
    handler.on_chain_start({"id": ["langgraph", "g", "G"]}, {}, run_id=root)
    handler.on_chain_error(ValueError("boom"), run_id=root)
    trace = tracing._get_last_trace()
    assert trace is not None
    assert trace["trace"]["status"] == "failed"
    assert any(s.get("exception_type") == "ValueError" for s in trace["spans"])


@pytest.mark.req("SDK-09")
def test_control_flow_in_tool_does_not_fail_tool_span():
    from langgraph.errors import GraphInterrupt

    handler = LangGraphCallbackHandler(agent_name="a")
    rid = uuid.uuid4()
    with tracing.run("a"):
        handler.on_tool_start({"name": "t"}, "in", run_id=rid)
        handler.on_tool_error(GraphInterrupt(), run_id=rid)
    tool_spans = [s for s in _spans() if s["type"] == "tool"]
    assert tool_spans
    assert all(
        s["status"] == "completed" and not s.get("exception_type") for s in tool_spans
    )
