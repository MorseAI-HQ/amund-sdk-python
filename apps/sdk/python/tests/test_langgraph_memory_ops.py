"""Tests for the LangGraph adapter v2 — memory ops (MEM-10).

Covers the Phase 9 surfaces extending the CTX-09 adapter:
  - BaseCheckpointSaver.put / get
  - BaseStore.put / search / delete
  - Summarization-node detection (consolidate sibling span + context_remediation)
  - 50ms dedup window
  - No-op outside trace context
  - CTX-09 backwards compat (memory_op_type="retrieve" set explicitly)

These tests mock LangGraph's checkpointer/store abstractions via sys.modules
injection so the adapter can be exercised end-to-end without pulling in the
real LangGraph dependency.
"""

from __future__ import annotations

import sys
import time
import types
import uuid
from typing import Any

import pytest

from morse_ai import tracing
from morse_ai.adapters.langgraph import (
    _MEMORY_OP_DEDUP_WINDOW_MS,
    LangGraphCallbackHandler,
    _emit_memory_op_span,
    _install_memory_op_patches,
    _patched_originals,
    _uninstall_memory_op_patches,
)

# ---------------------------------------------------------------------------
# Mock LangGraph surfaces — built via sys.modules injection so install() can
# import + patch them. We rebuild a fresh class hierarchy in each test.
# ---------------------------------------------------------------------------


class _FakeCheckpointSaver:
    """Stand-in for langgraph.checkpoint.base.BaseCheckpointSaver."""

    def __init__(self) -> None:
        self.put_calls: list[dict[str, Any]] = []
        self.get_calls: list[dict[str, Any]] = []

    def put(self, *args: Any, **kwargs: Any) -> Any:
        record = {"args": args, "kwargs": kwargs}
        self.put_calls.append(record)
        return {"checkpoint_id": "ck-1"}

    def get(self, *args: Any, **kwargs: Any) -> Any:
        record = {"args": args, "kwargs": kwargs}
        self.get_calls.append(record)
        return {"value": "snapshot"}


class _FakeStore:
    """Stand-in for langgraph.store.base.BaseStore."""

    def __init__(self) -> None:
        self.put_calls: list[Any] = []
        self.search_calls: list[Any] = []
        self.delete_calls: list[Any] = []

    def put(self, *args: Any, **kwargs: Any) -> Any:
        self.put_calls.append((args, kwargs))
        return None

    def search(self, *args: Any, **kwargs: Any) -> Any:
        self.search_calls.append((args, kwargs))
        return ["hit"]

    def delete(self, *args: Any, **kwargs: Any) -> Any:
        self.delete_calls.append((args, kwargs))
        return None


@pytest.fixture
def fake_langgraph_modules() -> Any:
    """Inject fake langgraph.checkpoint.base + langgraph.store.base into sys.modules.

    Returns a tuple of (CheckpointerClass, StoreClass) so tests can construct
    instances to call. The classes ARE the symbols the install() patcher
    looks up by name.

    Cleans up after the test — restores any pre-existing modules + drops
    the fake ones — even if the test fails.
    """
    originals: dict[str, types.ModuleType | None] = {
        "langgraph.checkpoint.base": sys.modules.get("langgraph.checkpoint.base"),
        "langgraph.store.base": sys.modules.get("langgraph.store.base"),
    }

    cp_mod = types.ModuleType("langgraph.checkpoint.base")
    cp_mod.BaseCheckpointSaver = _FakeCheckpointSaver
    sys.modules["langgraph.checkpoint.base"] = cp_mod

    store_mod = types.ModuleType("langgraph.store.base")
    store_mod.BaseStore = _FakeStore
    sys.modules["langgraph.store.base"] = store_mod

    yield _FakeCheckpointSaver, _FakeStore

    # Always uninstall any patches we may have set.
    _uninstall_memory_op_patches()
    for name, mod in originals.items():
        if mod is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = mod


# ---------------------------------------------------------------------------
# Mock helpers reused from CTX-09 tests
# ---------------------------------------------------------------------------


class FakeDoc:
    def __init__(self, content: str, metadata: dict[str, Any] | None = None) -> None:
        self.page_content = content
        self.metadata = metadata or {}


class MockLLMResult:
    def __init__(
        self,
        text: str = "response",
        completion_tokens: int | None = None,
    ) -> None:
        Gen = type("Gen", (), {"text": text})
        self.generations = [[Gen()]]
        self.llm_output: dict[str, Any] = {}
        if completion_tokens is not None:
            self.llm_output["token_usage"] = {
                "completion_tokens": completion_tokens,
                "prompt_tokens": 100,
                "total_tokens": 100 + completion_tokens,
            }


def _llm_serialized(name: str = "ChatOpenAI") -> dict[str, Any]:
    return {"id": ["langchain", "chat_models", name], "name": name}


def _chain_serialized_for_node(node_name: str) -> dict[str, Any]:
    return {"id": ["langgraph", "node", node_name], "name": node_name}


def _memory_spans(trace: dict[str, Any]) -> list[dict[str, Any]]:
    return [s for s in trace["spans"] if s.get("type") == "memory"]


def _llm_spans(trace: dict[str, Any]) -> list[dict[str, Any]]:
    return [s for s in trace["spans"] if s.get("type") == "llm"]


# ---------------------------------------------------------------------------
# Checkpointer / Store patching — happy paths via install()
# ---------------------------------------------------------------------------


@pytest.mark.req("MEM-10")
def test_checkpointer_put_emits_extract_memory_span(fake_langgraph_modules):
    _install_memory_op_patches()
    cp_cls, _ = fake_langgraph_modules
    cp = cp_cls()

    with tracing.run("test-agent"):
        cp.put({"k": 1})

    trace = tracing._get_last_trace()
    assert trace is not None
    mems = _memory_spans(trace)
    assert len(mems) == 1
    md = mems[0].get("metadata") or {}
    assert md.get("memory_op_type") == "extract"
    assert md.get("store_name") == "_FakeCheckpointSaver"
    assert md.get("memory_type") == "checkpoint"
    assert md.get("adapter") == "langgraph"
    # Original method still called.
    assert len(cp.put_calls) == 1


@pytest.mark.req("MEM-10")
def test_checkpointer_get_emits_retrieve_memory_span(fake_langgraph_modules):
    _install_memory_op_patches()
    cp_cls, _ = fake_langgraph_modules
    cp = cp_cls()

    with tracing.run("test-agent"):
        result = cp.get("key1")

    assert result == {"value": "snapshot"}
    trace = tracing._get_last_trace()
    assert trace is not None
    mems = _memory_spans(trace)
    assert len(mems) == 1
    md = mems[0].get("metadata") or {}
    assert md.get("memory_op_type") == "retrieve"
    assert md.get("store_name") == "_FakeCheckpointSaver"
    assert len(cp.get_calls) == 1


@pytest.mark.req("MEM-10")
def test_store_put_emits_extract_memory_span(fake_langgraph_modules):
    _install_memory_op_patches()
    _, store_cls = fake_langgraph_modules
    store = store_cls()

    with tracing.run("test-agent"):
        store.put(("ns",), "key1", {"v": 1})

    trace = tracing._get_last_trace()
    assert trace is not None
    mems = _memory_spans(trace)
    assert len(mems) == 1
    md = mems[0].get("metadata") or {}
    assert md.get("memory_op_type") == "extract"
    assert md.get("store_name") == "_FakeStore"
    assert md.get("memory_type") == "store"


@pytest.mark.req("MEM-10")
def test_store_search_emits_retrieve_memory_span(fake_langgraph_modules):
    _install_memory_op_patches()
    _, store_cls = fake_langgraph_modules
    store = store_cls()

    with tracing.run("test-agent"):
        result = store.search(("ns",), query="q")

    assert result == ["hit"]
    trace = tracing._get_last_trace()
    assert trace is not None
    mems = _memory_spans(trace)
    assert len(mems) == 1
    md = mems[0].get("metadata") or {}
    assert md.get("memory_op_type") == "retrieve"


@pytest.mark.req("MEM-10")
def test_store_delete_emits_prune_memory_span(fake_langgraph_modules):
    _install_memory_op_patches()
    _, store_cls = fake_langgraph_modules
    store = store_cls()

    with tracing.run("test-agent"):
        store.delete(("ns",), "key1")

    trace = tracing._get_last_trace()
    assert trace is not None
    mems = _memory_spans(trace)
    assert len(mems) == 1
    md = mems[0].get("metadata") or {}
    assert md.get("memory_op_type") == "prune"


# ---------------------------------------------------------------------------
# Multiple distinct ops within a single trace
# ---------------------------------------------------------------------------


@pytest.mark.req("MEM-10")
def test_multiple_distinct_ops_each_emit(fake_langgraph_modules):
    _install_memory_op_patches()
    cp_cls, store_cls = fake_langgraph_modules
    cp = cp_cls()
    store = store_cls()

    with tracing.run("test-agent"):
        cp.put({"k": 1})
        # Sleep just long enough to fall outside the dedup window even for
        # same-(store, op_type) combos.
        time.sleep(_MEMORY_OP_DEDUP_WINDOW_MS / 1000.0 + 0.01)
        cp.get("k")
        store.put((), "k", "v")
        store.delete((), "k")

    trace = tracing._get_last_trace()
    assert trace is not None
    mems = _memory_spans(trace)
    ops = sorted(m.get("metadata", {}).get("memory_op_type") for m in mems)
    assert ops == ["extract", "extract", "prune", "retrieve"]


# ---------------------------------------------------------------------------
# Dedup window — same (bucket, store, op_type) within 50ms collides
# ---------------------------------------------------------------------------


@pytest.mark.req("MEM-10")
def test_dedup_blocks_double_emit_within_50ms(fake_langgraph_modules):
    _install_memory_op_patches()
    cp_cls, _ = fake_langgraph_modules
    cp = cp_cls()

    with tracing.run("test-agent"):
        # Two back-to-back put() calls — guaranteed to land in the same
        # bucket since each call is sub-millisecond. The second emission
        # should be deduped.
        cp.put({"k": 1})
        cp.put({"k": 2})

    trace = tracing._get_last_trace()
    assert trace is not None
    mems = _memory_spans(trace)
    # Original method called twice (we never block user code).
    assert len(cp.put_calls) == 2
    # But only one memory span emitted.
    assert len(mems) == 1
    md = mems[0].get("metadata") or {}
    assert md.get("memory_op_type") == "extract"


@pytest.mark.req("MEM-10")
def test_dedup_resets_after_window(fake_langgraph_modules):
    """After the 50ms window passes, the same (store, op_type) emits again."""
    _install_memory_op_patches()
    cp_cls, _ = fake_langgraph_modules
    cp = cp_cls()

    with tracing.run("test-agent"):
        cp.put({"k": 1})
        # Sleep > 50ms (2x window for safety against scheduler jitter).
        time.sleep(_MEMORY_OP_DEDUP_WINDOW_MS / 1000.0 * 2.5)
        cp.put({"k": 2})

    trace = tracing._get_last_trace()
    assert trace is not None
    mems = _memory_spans(trace)
    assert len(mems) == 2


# ---------------------------------------------------------------------------
# No-op outside an active trace context
# ---------------------------------------------------------------------------


@pytest.mark.req("MEM-10")
def test_no_op_outside_trace_context(fake_langgraph_modules):
    """Calling patched methods with no active morse_ai.run yields no spans."""
    _install_memory_op_patches()
    cp_cls, store_cls = fake_langgraph_modules
    cp = cp_cls()
    store = store_cls()

    tracing._last_trace = None  # ensure stale state doesn't fool us

    # NOT inside tracing.run(...). All patched methods should be no-ops.
    cp.put({"k": 1})
    cp.get("k")
    store.put((), "k", "v")
    store.search(())
    store.delete((), "k")

    # No trace was sent because no trace was open.
    assert tracing._get_last_trace() is None
    # Original methods still called (we never block user code).
    assert len(cp.put_calls) == 1
    assert len(cp.get_calls) == 1
    assert len(store.put_calls) == 1
    assert len(store.search_calls) == 1
    assert len(store.delete_calls) == 1


# ---------------------------------------------------------------------------
# Original error propagation — we never swallow customer errors
# ---------------------------------------------------------------------------


@pytest.mark.req("MEM-10")
def test_wrapped_method_exception_propagates():
    """Customer-code errors must propagate AS-IS — wrapper never swallows them."""

    # Build a fresh fake checkpointer whose put() raises, install patches
    # for THIS class hierarchy specifically, then call.
    class RaisingCheckpointSaver:
        def put(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("user error")

        def get(self, *args: Any, **kwargs: Any) -> Any:
            return None

    cp_mod = types.ModuleType("langgraph.checkpoint.base")
    cp_mod.BaseCheckpointSaver = RaisingCheckpointSaver
    store_mod = types.ModuleType("langgraph.store.base")
    store_mod.BaseStore = _FakeStore

    original_cp_mod = sys.modules.get("langgraph.checkpoint.base")
    original_store_mod = sys.modules.get("langgraph.store.base")
    sys.modules["langgraph.checkpoint.base"] = cp_mod
    sys.modules["langgraph.store.base"] = store_mod
    try:
        _install_memory_op_patches()
        cp = RaisingCheckpointSaver()
        with tracing.run("test-agent"):  # noqa: SIM117
            with pytest.raises(RuntimeError, match="user error"):
                cp.put({"k": 1})
        # AND: because the wrapped method raised, no memory span should have
        # been emitted (emission happens only after a successful original call).
        trace = tracing._get_last_trace()
        assert trace is not None
        assert _memory_spans(trace) == []
    finally:
        _uninstall_memory_op_patches()
        if original_cp_mod is None:
            sys.modules.pop("langgraph.checkpoint.base", None)
        else:
            sys.modules["langgraph.checkpoint.base"] = original_cp_mod
        if original_store_mod is None:
            sys.modules.pop("langgraph.store.base", None)
        else:
            sys.modules["langgraph.store.base"] = original_store_mod


# ---------------------------------------------------------------------------
# Idempotent install — calling install twice doesn't double-wrap
# ---------------------------------------------------------------------------


@pytest.mark.req("MEM-10")
def test_install_idempotent(fake_langgraph_modules):
    _install_memory_op_patches()
    first = dict(_patched_originals)
    _install_memory_op_patches()  # second call must NOT re-wrap
    second = dict(_patched_originals)
    # Same set of patched keys; same original references stored.
    assert first.keys() == second.keys()
    for k in first:
        assert first[k] is second[k]


@pytest.mark.req("MEM-10")
def test_uninstall_restores_originals(fake_langgraph_modules):
    cp_cls, store_cls = fake_langgraph_modules
    original_put = cp_cls.put
    _install_memory_op_patches()
    # After install, the class method is the wrapped version.
    assert cp_cls.put is not original_put
    _uninstall_memory_op_patches()
    # After uninstall, restored.
    assert cp_cls.put is original_put
    assert not _patched_originals


# ---------------------------------------------------------------------------
# Summarization-node detection
# ---------------------------------------------------------------------------


@pytest.mark.req("MEM-10")
def test_summarize_node_llm_emits_consolidate_sibling_span():
    """LLM call inside a `summarize_*` node → sibling memory span op=consolidate."""
    handler = LangGraphCallbackHandler("test-agent")
    chain_run = uuid.uuid4()
    llm_run = uuid.uuid4()

    with tracing.run("test-agent"):
        handler.on_chain_start(
            _chain_serialized_for_node("summarize_history"),
            {},
            run_id=chain_run,
            metadata={"langgraph_node": "summarize_history"},
        )
        handler.on_chat_model_start(
            _llm_serialized(),
            [],
            run_id=llm_run,
        )
        handler.on_llm_end(MockLLMResult(completion_tokens=42), run_id=llm_run)
        handler.on_chain_end({}, run_id=chain_run)

    trace = tracing._get_last_trace()
    assert trace is not None
    mems = _memory_spans(trace)
    llms = _llm_spans(trace)
    assert len(llms) == 1
    consolidate = [
        m for m in mems if m.get("metadata", {}).get("memory_op_type") == "consolidate"
    ]
    assert len(consolidate) == 1
    md = consolidate[0].get("metadata") or {}
    assert md.get("producer_span_id") == llms[0]["span_id"]
    assert md.get("node_name") == "summarize_history"
    assert md.get("summary_token_count") == 42


@pytest.mark.req("MEM-10")
def test_summarize_node_sets_context_remediation_on_next_llm():
    """LLM call AFTER a summarize_* node closes carries metadata.context_remediation."""
    handler = LangGraphCallbackHandler("test-agent")
    summarize_chain = uuid.uuid4()
    summarize_llm = uuid.uuid4()
    next_llm = uuid.uuid4()

    with tracing.run("test-agent"):
        handler.on_chain_start(
            _chain_serialized_for_node("summarize_history"),
            {},
            run_id=summarize_chain,
            metadata={"langgraph_node": "summarize_history"},
        )
        handler.on_chat_model_start(_llm_serialized(), [], run_id=summarize_llm)
        handler.on_llm_end(MockLLMResult(), run_id=summarize_llm)
        handler.on_chain_end({}, run_id=summarize_chain)
        # Post-summarization LLM call — should carry remediation.
        handler.on_chat_model_start(_llm_serialized(), [], run_id=next_llm)
        handler.on_llm_end(MockLLMResult(), run_id=next_llm)

    trace = tracing._get_last_trace()
    assert trace is not None
    llms = _llm_spans(trace)
    assert len(llms) == 2
    # Insertion order: summarize_llm first (added to completed_spans on close),
    # then next_llm. The remediation lands on the SECOND LLM.
    summarize_span = llms[0]
    next_span = llms[1]
    sum_remediation = (summarize_span.get("metadata") or {}).get("context_remediation")
    next_remediation = (next_span.get("metadata") or {}).get("context_remediation")
    assert sum_remediation is None, "summarize-call LLM must not carry remediation"
    assert next_remediation is not None
    assert next_remediation["strategy"] == "summarize"
    assert next_remediation["remediation_metadata"]["node_name"] == "summarize_history"


@pytest.mark.req("MEM-10")
def test_consolidate_node_name_also_triggers():
    """Node names matching `consolidate*` also flag summarization."""
    handler = LangGraphCallbackHandler("test-agent")
    chain_run = uuid.uuid4()
    llm_run = uuid.uuid4()

    with tracing.run("test-agent"):
        handler.on_chain_start(
            _chain_serialized_for_node("consolidate_memory"),
            {},
            run_id=chain_run,
            metadata={"langgraph_node": "consolidate_memory"},
        )
        handler.on_chat_model_start(_llm_serialized(), [], run_id=llm_run)
        handler.on_llm_end(MockLLMResult(), run_id=llm_run)
        handler.on_chain_end({}, run_id=chain_run)

    trace = tracing._get_last_trace()
    assert trace is not None
    mems = _memory_spans(trace)
    consolidate = [
        m for m in mems if m.get("metadata", {}).get("memory_op_type") == "consolidate"
    ]
    assert len(consolidate) == 1


@pytest.mark.req("MEM-10")
def test_non_summarize_node_does_not_trigger():
    """Regular nodes (e.g., `retrieve_docs`) must NOT emit consolidate spans."""
    handler = LangGraphCallbackHandler("test-agent")
    chain_run = uuid.uuid4()
    llm_run = uuid.uuid4()

    with tracing.run("test-agent"):
        handler.on_chain_start(
            _chain_serialized_for_node("retrieve_docs"),
            {},
            run_id=chain_run,
            metadata={"langgraph_node": "retrieve_docs"},
        )
        handler.on_chat_model_start(_llm_serialized(), [], run_id=llm_run)
        handler.on_llm_end(MockLLMResult(), run_id=llm_run)
        handler.on_chain_end({}, run_id=chain_run)

    trace = tracing._get_last_trace()
    assert trace is not None
    mems = _memory_spans(trace)
    consolidate = [
        m for m in mems if m.get("metadata", {}).get("memory_op_type") == "consolidate"
    ]
    assert consolidate == []


@pytest.mark.req("MEM-10")
def test_summarization_case_insensitive():
    """Node name pattern is case-insensitive."""
    handler = LangGraphCallbackHandler("test-agent")
    chain_run = uuid.uuid4()
    llm_run = uuid.uuid4()

    with tracing.run("test-agent"):
        handler.on_chain_start(
            _chain_serialized_for_node("Summarize_Stuff"),
            {},
            run_id=chain_run,
            metadata={"langgraph_node": "Summarize_Stuff"},
        )
        handler.on_chat_model_start(_llm_serialized(), [], run_id=llm_run)
        handler.on_llm_end(MockLLMResult(), run_id=llm_run)
        handler.on_chain_end({}, run_id=chain_run)

    trace = tracing._get_last_trace()
    assert trace is not None
    mems = _memory_spans(trace)
    assert any(
        m.get("metadata", {}).get("memory_op_type") == "consolidate" for m in mems
    )


# ---------------------------------------------------------------------------
# _emit_memory_op_span direct unit test — no-op outside trace
# ---------------------------------------------------------------------------


@pytest.mark.req("MEM-10")
def test_emit_memory_op_span_no_op_outside_trace():
    """Calling the emission helper outside a trace returns None and emits nothing."""
    tracing._last_trace = None
    result = _emit_memory_op_span(
        name="x",
        memory_op_type="retrieve",
        store_name="foo",
    )
    assert result is None
    assert tracing._get_last_trace() is None


# ---------------------------------------------------------------------------
# CTX-09 backwards compat — retriever spans now carry memory_op_type explicitly
# ---------------------------------------------------------------------------


@pytest.mark.req("MEM-10", "CTX-09")
def test_ctx09_retriever_span_now_carries_explicit_memory_op_type():
    """CTX-09 retriever-driven memory span carries memory_op_type='retrieve'.

    The server's classifier already defaults missing memory_op_type to
    'retrieve', so this is a no-op for storage — but explicit is better
    than implicit, and the spec requires it for MEM-10 (set if missing).
    """
    handler = LangGraphCallbackHandler("test-agent")
    rid = uuid.uuid4()

    with tracing.run("test-agent"):
        handler.on_retriever_start(
            {"id": ["langchain_community", "vectorstores", "FAISS"], "name": "FAISS"},
            "what?",
            run_id=rid,
        )
        handler.on_retriever_end([FakeDoc("alpha", {"source": "a"})], run_id=rid)

    trace = tracing._get_last_trace()
    assert trace is not None
    mems = _memory_spans(trace)
    assert len(mems) == 1
    md = mems[0].get("metadata") or {}
    assert md.get("memory_op_type") == "retrieve"
    # CTX-09 fields preserved.
    assert md.get("memory_type") == "vector"
    assert md.get("store_name") == "FAISS"
    assert md.get("top_k") == 1


# ---------------------------------------------------------------------------
# Async checkpoint patching — Bug 1 fix
# LangGraph calls aput / aget_tuple / aput_writes (not put/get) when the
# graph runs in an async context (FastAPI, astream, ainvoke).
# ---------------------------------------------------------------------------


class _FakeCheckpointSaverWithAsync(_FakeCheckpointSaver):
    """Extends fake saver with async methods matching BaseCheckpointSaver."""

    async def aput(self, *args: Any, **kwargs: Any) -> Any:
        self.put_calls.append({"args": args, "kwargs": kwargs, "async": True})
        return {"checkpoint_id": "ck-async-1"}

    async def aput_writes(self, *args: Any, **kwargs: Any) -> Any:
        self.put_calls.append({"args": args, "kwargs": kwargs, "async_writes": True})
        return None

    async def aget_tuple(self, *args: Any, **kwargs: Any) -> Any:
        self.get_calls.append({"args": args, "kwargs": kwargs, "async": True})
        return {"value": "async-snapshot"}


@pytest.fixture
def fake_async_langgraph_modules() -> Any:
    """Inject fake modules with async-capable checkpointer into sys.modules."""
    originals: dict[str, types.ModuleType | None] = {
        "langgraph.checkpoint.base": sys.modules.get("langgraph.checkpoint.base"),
        "langgraph.store.base": sys.modules.get("langgraph.store.base"),
    }

    cp_mod = types.ModuleType("langgraph.checkpoint.base")
    cp_mod.BaseCheckpointSaver = _FakeCheckpointSaverWithAsync
    sys.modules["langgraph.checkpoint.base"] = cp_mod

    store_mod = types.ModuleType("langgraph.store.base")
    store_mod.BaseStore = _FakeStore
    sys.modules["langgraph.store.base"] = store_mod

    yield _FakeCheckpointSaverWithAsync, _FakeStore

    _uninstall_memory_op_patches()
    for name, mod in originals.items():
        if mod is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = mod


@pytest.mark.req("MEM-10")
def test_async_aput_emits_extract_memory_span(fake_async_langgraph_modules):
    """BaseCheckpointSaver.aput must emit a memory span on the async execution path.

    LangGraph calls aput (not put) when the graph is invoked via astream/ainvoke
    inside an async framework (FastAPI, etc.). Without patching aput, InMemorySaver
    and PostgresSaver both produce zero memory spans for async-mode graphs.
    """
    import asyncio

    _install_memory_op_patches()
    cp_cls, _ = fake_async_langgraph_modules
    cp = cp_cls()

    async def _run():
        with tracing.run("async-test-agent"):
            await cp.aput("config", "checkpoint", {"k": "v"}, [])

    asyncio.run(_run())

    trace = tracing._get_last_trace()
    assert trace is not None
    mems = _memory_spans(trace)
    assert len(mems) == 1, f"Expected 1 memory span from aput, got {len(mems)}"
    md = mems[0].get("metadata") or {}
    assert md.get("memory_op_type") == "extract"
    assert md.get("memory_type") == "checkpoint"
    assert md.get("async") is True


@pytest.mark.req("MEM-10")
def test_async_aget_tuple_emits_retrieve_memory_span(fake_async_langgraph_modules):
    """aget_tuple must emit a memory span (async checkpoint read)."""
    import asyncio

    _install_memory_op_patches()
    cp_cls, _ = fake_async_langgraph_modules
    cp = cp_cls()

    async def _run():
        with tracing.run("async-test-agent"):
            await cp.aget_tuple({"configurable": {"thread_id": "t1"}})

    asyncio.run(_run())

    trace = tracing._get_last_trace()
    assert trace is not None
    mems = _memory_spans(trace)
    assert len(mems) == 1, f"Expected 1 memory span from aget_tuple, got {len(mems)}"
    md = mems[0].get("metadata") or {}
    assert md.get("memory_op_type") == "retrieve"
    assert md.get("memory_type") == "checkpoint"


@pytest.mark.req("MEM-10")
def test_async_aput_writes_emits_extract_memory_span(fake_async_langgraph_modules):
    """aput_writes must emit a memory span (async incremental write)."""
    import asyncio

    _install_memory_op_patches()
    cp_cls, _ = fake_async_langgraph_modules
    cp = cp_cls()

    async def _run():
        with tracing.run("async-test-agent"):
            await cp.aput_writes("config", [("channel", "value")], "task-id")

    asyncio.run(_run())

    trace = tracing._get_last_trace()
    assert trace is not None
    mems = _memory_spans(trace)
    assert len(mems) == 1
    assert mems[0].get("metadata", {}).get("memory_op_type") == "extract"


@pytest.mark.req("MEM-10")
def test_async_checkpoint_noop_outside_trace(fake_async_langgraph_modules):
    """aput/aget_tuple are no-ops when no trace is active — no span emitted."""
    import asyncio

    _install_memory_op_patches()
    cp_cls, _ = fake_async_langgraph_modules
    cp = cp_cls()

    # Record the last-trace pointer before the calls so we can confirm
    # these calls do NOT add a new trace or memory spans.
    trace_before = tracing._get_last_trace()

    async def _run():
        await cp.aput("config", "checkpoint", {}, [])
        await cp.aget_tuple({"configurable": {}})

    asyncio.run(_run())

    # The last trace must not have changed — no new trace was recorded.
    assert tracing._get_last_trace() is trace_before


# ---------------------------------------------------------------------------
# Sub-agent correlation — Bug 3 fix
# When a sub-agent compiled graph fires on_chain_start(parent_run_id=None)
# inside an existing active AV trace, it must inherit the parent's trace
# rather than opening a new root trace.
# ---------------------------------------------------------------------------


@pytest.mark.req("MEM-10")
def test_sub_agent_inherits_parent_trace_context():
    """LangGraphCallbackHandler._ensure_trace must reuse an existing _active_trace.

    Reproduces the multi-agent pair pattern: supervisor graph opens trace A,
    then a sub-agent's graph starts with parent_run_id=None inside the same
    asyncio task. Without the fix, _ensure_trace creates trace B. With the
    fix, the sub-agent handler sees _active_trace is already set and returns
    without creating a new trace — its _auto_trace stays False so it never
    closes the parent trace when its own chain ends.
    """
    from morse_ai.tracing import _active_trace

    sub_handler = LangGraphCallbackHandler("sub-agent")

    # Simulate the parent trace being active (set by the supervisor's
    # on_chain_start path which calls _ensure_trace → run().__enter__).
    with tracing.run("supervisor-agent"):
        # _active_trace is now set. A sub-agent's on_chain_start fires here.
        assert _active_trace.get() is not None, "Parent trace must be active"

        # Sub-agent calls _ensure_trace — should NOT create a new trace.
        sub_handler._ensure_trace()

        # _auto_trace should be False: the sub-agent isn't the trace owner.
        assert sub_handler._auto_trace is False, (
            "Sub-agent must not mark itself as auto_trace owner — it would "
            "close the parent trace when its chain ends"
        )
        # _run_context must be None: sub-agent didn't open a new TracedRun.
        assert sub_handler._run_context is None, (
            "Sub-agent must not open its own TracedRun (that would be a new root trace)"
        )

    # After context exits, verify only ONE trace was recorded (the parent's).
    trace = tracing._get_last_trace()
    assert trace is not None


@pytest.mark.req("MEM-10")
def test_independent_agent_gets_own_trace_when_no_active_context():
    """_ensure_trace creates a new trace when no parent trace is active.

    Regression guard: we must NOT suppress new traces when there's genuinely
    no parent running (e.g., a fresh standalone invocation in a new async task).
    """
    handler = LangGraphCallbackHandler("standalone-agent")

    from morse_ai.tracing import _active_trace

    assert _active_trace.get() is None, "No active trace should exist before this test"

    handler._ensure_trace()

    # Should have opened its own trace.
    assert handler._auto_trace is True
    assert handler._run_context is not None

    # Cleanup.
    handler._run_context.__exit__(None, None, None)
    handler._run_context = None
    handler._auto_trace = False
