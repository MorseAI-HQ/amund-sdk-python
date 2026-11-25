"""Tests for the LangGraph adapter's memory-span + context_segments wiring.

These tests mock LangChain Document objects and fire callbacks directly on
the LangGraphCallbackHandler — they do NOT require langchain or langgraph
installed. The adapter module guards its `install()` path behind
`import langgraph`; constructing the callback handler is always safe.
"""

# ruff: noqa: E501, SIM117
from __future__ import annotations

import hashlib
import uuid
from typing import Any

import pytest

from amund import tracing
from amund.adapters.langgraph import (
    _SEGMENT_CONTENT_MAX_BYTES,
    LangGraphCallbackHandler,
)

# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


class FakeDoc:
    """Minimal duck-typed LangChain Document — page_content + metadata."""

    def __init__(self, content: str, metadata: dict[str, Any] | None = None) -> None:
        self.page_content = content
        self.metadata = metadata or {}


class MockLLMResult:
    """Minimal mock for LangChain LLMResult — used to close LLM spans."""

    def __init__(self, text: str = "response", tokens: dict | None = None) -> None:
        Gen = type("Gen", (), {"text": text})
        self.generations = [[Gen()]]
        self.llm_output = tokens or {}


def _retriever_serialized(name: str = "FAISS") -> dict[str, Any]:
    return {
        "id": ["langchain_community", "vectorstores", name],
        "name": name,
    }


def _llm_serialized(name: str = "ChatOpenAI") -> dict[str, Any]:
    return {"id": ["langchain", "chat_models", name], "name": name}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.req("CTX-09")
def test_retriever_emits_memory_span():
    """on_retriever_start + on_retriever_end produce a memory-typed span."""
    handler = LangGraphCallbackHandler("test-agent")
    run_id = uuid.uuid4()
    docs = [
        FakeDoc("alpha", {"source": "a"}),
        FakeDoc("beta", {"source": "b"}),
        FakeDoc("gamma", {"source": "c"}),
    ]

    with tracing.run("test-agent"):
        handler.on_retriever_start(
            _retriever_serialized("FAISS"), "what?", run_id=run_id
        )
        handler.on_retriever_end(docs, run_id=run_id)

    trace = tracing._get_last_trace()
    assert trace is not None
    memory_spans = [s for s in trace["spans"] if s.get("type") == "memory"]
    assert len(memory_spans) == 1
    span = memory_spans[0]
    assert span["status"] == "completed"
    md = span.get("metadata") or {}
    assert md.get("memory_type") == "vector"
    assert md.get("store_name") == "FAISS"
    assert md.get("top_k") == 3
    out = span.get("output_data") or {}
    assert "documents" in out
    assert len(out["documents"]) == 3
    assert out["documents"][0]["content"] == "alpha"
    assert out["documents"][0]["metadata"] == {"source": "a"}
    # RAGQ-01: same ranked documents[] shape as the LangChain retriever span.
    assert [d["rank"] for d in out["documents"]] == [1, 2, 3]
    assert out["documents"][0]["source"] == "a"
    assert out["documents"][0]["id"]


@pytest.mark.req("CTX-09")
def test_langgraph_root_callback_restores_parent_trace_after_contextvar_loss():
    """Root graph callbacks inherit an explicit parent run across context loss."""
    chain_run = uuid.uuid4()
    llm_run = uuid.uuid4()

    with tracing.run("parent-graph"):
        handler = LangGraphCallbackHandler("detached-graph")
        token = tracing._active_trace.set(None)
        try:
            handler.on_chain_start(
                {"id": ["langgraph", "graph", "CompiledStateGraph"]},
                {"ticker": "AAPL"},
                run_id=chain_run,
            )
            handler.on_chat_model_start(
                _llm_serialized("ChatOpenAI"),
                [],
                run_id=llm_run,
                parent_run_id=chain_run,
                invocation_params={"model_name": "gpt-4o-mini"},
            )
            handler.on_llm_end(
                MockLLMResult(
                    tokens={
                        "token_usage": {
                            "prompt_tokens": 3,
                            "completion_tokens": 5,
                            "total_tokens": 8,
                        }
                    }
                ),
                run_id=llm_run,
            )
            handler.on_chain_end({"decision": "buy"}, run_id=chain_run)
        finally:
            tracing._active_trace.reset(token)

    trace = tracing._get_last_trace()
    assert trace is not None
    assert trace["trace"]["agent"] == "parent-graph"
    assert handler._run_context is None

    llm_spans = [s for s in trace["spans"] if s.get("type") == "llm"]
    assert len(llm_spans) == 1
    assert llm_spans[0]["total_tokens"] == 8


@pytest.mark.req("CTX-09")
def test_subsequent_llm_span_has_memory_vector_segments():
    """Retriever output → context_segments on the next LLM span, with producer_span_id link."""
    handler = LangGraphCallbackHandler("test-agent")
    retriever_run = uuid.uuid4()
    llm_run = uuid.uuid4()
    docs = [
        FakeDoc("first chunk", {"source": "doc1.md"}),
        FakeDoc("second chunk", {"source": "doc2.md"}),
    ]

    with tracing.run("test-agent"):
        handler.on_retriever_start(_retriever_serialized(), "q", run_id=retriever_run)
        handler.on_retriever_end(docs, run_id=retriever_run)
        handler.on_chat_model_start(
            _llm_serialized("ChatOpenAI"),
            [],
            run_id=llm_run,
            invocation_params={"model_name": "gpt-4o"},
        )
        handler.on_llm_end(MockLLMResult(), run_id=llm_run)

    trace = tracing._get_last_trace()
    assert trace is not None
    memory_spans = [s for s in trace["spans"] if s.get("type") == "memory"]
    llm_spans = [s for s in trace["spans"] if s.get("type") == "llm"]
    assert len(memory_spans) == 1
    assert len(llm_spans) == 1
    memory_span_id = memory_spans[0]["span_id"]

    segments = llm_spans[0].get("context_segments") or []
    memory_vector = [s for s in segments if s.get("source") == "memory_vector"]
    assert len(memory_vector) == 2
    for seg in memory_vector:
        assert seg.get("producer_span_id") == memory_span_id


@pytest.mark.req("CTX-09")
def test_memory_vector_segments_have_detection_confidence_exact():
    """Adapter knows exactly which chunks were retrieved → attrs.detection_confidence == 'exact'."""
    handler = LangGraphCallbackHandler("test-agent")
    rid = uuid.uuid4()
    lid = uuid.uuid4()

    with tracing.run("test-agent"):
        handler.on_retriever_start(_retriever_serialized(), "q", run_id=rid)
        handler.on_retriever_end([FakeDoc("c1")], run_id=rid)
        handler.on_chat_model_start(_llm_serialized(), [], run_id=lid)
        handler.on_llm_end(MockLLMResult(), run_id=lid)

    trace = tracing._get_last_trace()
    assert trace is not None
    llm_spans = [s for s in trace["spans"] if s.get("type") == "llm"]
    assert len(llm_spans) == 1
    segs = llm_spans[0].get("context_segments") or []
    assert segs, "expected at least one context segment"
    for s in segs:
        attrs = s.get("attrs") or {}
        assert attrs.get("detection_confidence") == "exact"


@pytest.mark.req("CTX-09")
def test_pending_docs_consumed_after_one_llm_call():
    """retrieve → LLM → LLM: only the first LLM gets the memory_vector segments."""
    handler = LangGraphCallbackHandler("test-agent")
    rid = uuid.uuid4()
    lid1 = uuid.uuid4()
    lid2 = uuid.uuid4()

    with tracing.run("test-agent"):
        handler.on_retriever_start(_retriever_serialized(), "q", run_id=rid)
        handler.on_retriever_end([FakeDoc("c1"), FakeDoc("c2")], run_id=rid)
        # First LLM should consume.
        handler.on_chat_model_start(_llm_serialized(), [], run_id=lid1)
        handler.on_llm_end(MockLLMResult(), run_id=lid1)
        # Second LLM should NOT see memory_vector segments.
        handler.on_chat_model_start(_llm_serialized(), [], run_id=lid2)
        handler.on_llm_end(MockLLMResult(), run_id=lid2)

    trace = tracing._get_last_trace()
    assert trace is not None
    llm_spans = [s for s in trace["spans"] if s.get("type") == "llm"]
    assert len(llm_spans) == 2
    # Order in trace["spans"] is insertion order; first LLM is the first one.
    first_segs = llm_spans[0].get("context_segments") or []
    second_segs = llm_spans[1].get("context_segments") or []
    first_mem = [s for s in first_segs if s.get("source") == "memory_vector"]
    second_mem = [s for s in second_segs if s.get("source") == "memory_vector"]
    assert len(first_mem) == 2
    assert len(second_mem) == 0


@pytest.mark.req("CTX-09")
def test_retriever_without_end_does_not_leak_state():
    """on_retriever_start without on_retriever_end → trace finalizes cleanly, no orphan span, no exception."""
    handler = LangGraphCallbackHandler("test-agent")
    rid = uuid.uuid4()

    # Should not raise.
    with tracing.run("test-agent"):
        handler.on_retriever_start(_retriever_serialized(), "q", run_id=rid)
        # ...nothing — trace ends mid-retrieve

    trace = tracing._get_last_trace()
    assert trace is not None
    # The trace should NOT contain a 'completed' memory span without
    # output_data. Because on_retriever_end never fired, the open memory
    # span is still in handler._span_stack — it never made it into
    # state.completed_spans, so it's absent from the wire payload.
    memory_spans = [s for s in trace["spans"] if s.get("type") == "memory"]
    assert memory_spans == []


@pytest.mark.req("CTX-09")
def test_retriever_outside_trace_no_ops():
    """Callbacks fired with no active trace → no exception, no spans."""
    handler = LangGraphCallbackHandler("no-trace-agent")
    rid = uuid.uuid4()
    lid = uuid.uuid4()

    # Reset last_trace so a stale value from a previous test doesn't fool us.
    tracing._last_trace = None

    # No tracing.run(...) wrapper — auto-trace is the LangChain handler's
    # default for chains, but for retrievers the adapter explicitly no-ops
    # outside an active trace context.
    handler.on_retriever_start(_retriever_serialized(), "q", run_id=rid)
    handler.on_retriever_end([FakeDoc("x")], run_id=rid)

    # No trace was created.
    assert tracing._get_last_trace() is None

    # Same for an LLM start with no pending state.
    handler.on_chat_model_start(_llm_serialized(), [], run_id=lid)
    # LLM start auto-creates a chain trace via parent — close it cleanly.
    handler.on_llm_end(MockLLMResult(), run_id=lid)


@pytest.mark.req("CTX-09")
def test_long_document_content_truncated():
    """A 5000-byte doc → segment.content_truncated True; content_hash covers FULL pre-truncation bytes."""
    handler = LangGraphCallbackHandler("test-agent")
    rid = uuid.uuid4()
    lid = uuid.uuid4()
    big = "x" * 5000  # 5000 ASCII bytes
    full_hash = hashlib.sha256(big.encode("utf-8")).hexdigest()

    with tracing.run("test-agent"):
        handler.on_retriever_start(_retriever_serialized(), "q", run_id=rid)
        handler.on_retriever_end([FakeDoc(big, {"source": "big.md"})], run_id=rid)
        handler.on_chat_model_start(_llm_serialized(), [], run_id=lid)
        handler.on_llm_end(MockLLMResult(), run_id=lid)

    trace = tracing._get_last_trace()
    assert trace is not None
    llm_spans = [s for s in trace["spans"] if s.get("type") == "llm"]
    segs = [
        s
        for s in (llm_spans[0].get("context_segments") or [])
        if s.get("source") == "memory_vector"
    ]
    assert len(segs) == 1
    seg = segs[0]
    assert seg["content_truncated"] is True
    # Byte length of stored content respects the cap.
    assert len(seg["content"].encode("utf-8")) <= _SEGMENT_CONTENT_MAX_BYTES
    # Hash is over the full pre-truncation content.
    assert seg["content_hash"] == full_hash


@pytest.mark.req("CTX-09")
def test_callbacks_swallow_errors(monkeypatch):
    """If SpanData construction explodes mid-callback, the callback returns None without raising."""
    handler = LangGraphCallbackHandler("test-agent")
    rid = uuid.uuid4()

    import amund.tracing as tracing_module

    class Boom:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("simulated SpanData failure")

    # Enter the run() context FIRST (uses real SpanData), then patch
    # SpanData so child-span construction inside the callbacks blows up.
    with tracing.run("test-agent"):
        monkeypatch.setattr(tracing_module, "SpanData", Boom)
        result_start = handler.on_retriever_start(
            _retriever_serialized(), "q", run_id=rid
        )
        result_end = handler.on_retriever_end([FakeDoc("c")], run_id=rid)
        assert result_start is None
        assert result_end is None
        # Restore before exiting the context manager so __exit__ doesn't
        # itself try to build a SpanData via Boom.
        monkeypatch.undo()


@pytest.mark.req("CTX-09")
def test_langgraph_memory_scope_on_extract(monkeypatch):
    """Checkpointer/store extract spans carry memory_scope='working'."""
    import amund.adapters.langgraph as _lg
    import amund.tracing as _t
    from amund.tracing import _active_trace, _TraceState

    captured = []
    orig = _t._finalize_span

    def _spy(span, start, **kwargs):
        orig(span, start, **kwargs)
        captured.append(span)

    # Patch in the adapter module's namespace (it imported _finalize_span directly)
    monkeypatch.setattr(_lg, "_finalize_span", _spy)

    emit_fn = getattr(_lg, "_emit_memory_op_span", None)
    assert emit_fn is not None, "Could not find _emit_memory_op_span"

    state = _TraceState(trace_id="test-scope", agent_name="test-agent")
    token = _active_trace.set(state)
    try:
        emit_fn(
            name="checkpointer.InMemorySaver.put",
            store_name="langgraph_checkpointer",
            memory_op_type="extract",
        )
    finally:
        _active_trace.reset(token)

    mem = [s for s in captured if getattr(s, "type", None) == "memory"]
    assert mem, "No memory span emitted"
    assert mem[0].metadata.get("memory_scope") == "working"


@pytest.mark.req("CTX-09")
def test_langgraph_memory_scope_on_store_search(monkeypatch):
    """Store search spans carry memory_scope='retrieval'."""
    import amund.adapters.langgraph as _lg
    import amund.tracing as _t
    from amund.tracing import _active_trace, _TraceState

    captured = []
    orig = _t._finalize_span

    def _spy(span, start, **kwargs):
        orig(span, start, **kwargs)
        captured.append(span)

    monkeypatch.setattr(_lg, "_finalize_span", _spy)

    emit_fn = getattr(_lg, "_emit_memory_op_span", None)
    assert emit_fn is not None

    state = _TraceState(trace_id="test-scope-2", agent_name="test-agent")
    token = _active_trace.set(state)
    try:
        emit_fn(
            name="store.InMemoryStore.search",
            store_name="langgraph_store",
            memory_op_type="retrieve",
            is_store_search=True,
        )
    finally:
        _active_trace.reset(token)

    mem = [s for s in captured if getattr(s, "type", None) == "memory"]
    assert mem, "No memory span emitted"
    assert mem[0].metadata.get("memory_scope") == "retrieval"


@pytest.mark.req("CTX-09")
def test_langgraph_memory_scope_on_consolidate(monkeypatch):
    """Consolidate spans carry memory_scope='agent'."""
    import amund.adapters.langgraph as _lg
    import amund.tracing as _t
    from amund.tracing import _active_trace, _TraceState

    captured = []
    orig = _t._finalize_span

    def _spy(span, start, **kwargs):
        orig(span, start, **kwargs)
        captured.append(span)

    monkeypatch.setattr(_lg, "_finalize_span", _spy)

    emit_fn = getattr(_lg, "_emit_memory_op_span", None)
    assert emit_fn is not None

    state = _TraceState(trace_id="test-scope-3", agent_name="test-agent")
    token = _active_trace.set(state)
    try:
        emit_fn(
            name="summarization.summarize_node",
            store_name="langgraph_node:summarize_node",
            memory_op_type="consolidate",
        )
    finally:
        _active_trace.reset(token)

    mem = [s for s in captured if getattr(s, "type", None) == "memory"]
    assert mem, "No memory span emitted"
    assert mem[0].metadata.get("memory_scope") == "agent"


@pytest.mark.req("CTX-09")
def test_segment_position_is_sequential():
    """Multiple retrieved chunks → positions 0, 1, 2, ... on the LLM span."""
    handler = LangGraphCallbackHandler("test-agent")
    rid = uuid.uuid4()
    lid = uuid.uuid4()

    with tracing.run("test-agent"):
        handler.on_retriever_start(_retriever_serialized(), "q", run_id=rid)
        handler.on_retriever_end(
            [FakeDoc("a"), FakeDoc("b"), FakeDoc("c"), FakeDoc("d")],
            run_id=rid,
        )
        handler.on_chat_model_start(_llm_serialized(), [], run_id=lid)
        handler.on_llm_end(MockLLMResult(), run_id=lid)

    trace = tracing._get_last_trace()
    assert trace is not None
    llm_spans = [s for s in trace["spans"] if s.get("type") == "llm"]
    segs = [
        s
        for s in (llm_spans[0].get("context_segments") or [])
        if s.get("source") == "memory_vector"
    ]
    positions = [s["position"] for s in segs]
    assert positions == [0, 1, 2, 3]
