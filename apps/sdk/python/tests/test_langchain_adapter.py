"""Tests for the LangChain callback handler adapter.

These tests do NOT require langchain installed — handler methods are called
directly with mock arguments. Uses _get_last_trace() to inspect span output.
"""

# ruff: noqa: E501, SIM117
from __future__ import annotations

import uuid

import pytest

from morse_ai import tracing
from morse_ai.adapters.langchain import MorseCallbackHandler

# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


class MockLLMResult:
    """Minimal mock for LangChain LLMResult."""

    def __init__(self, text: str = "response text", tokens: dict | None = None):
        Gen = type("Gen", (), {"text": text})
        self.generations = [[Gen()]]
        self.llm_output = tokens or {}


def _chain_serialized(name: str = "MyChain") -> dict:
    return {"id": ["langchain", "chains", name], "name": name}


def _llm_serialized(name: str = "FakeLLM") -> dict:
    return {"id": ["langchain", "llms", name], "name": name}


def _tool_serialized(name: str = "my_tool") -> dict:
    return {"id": ["langchain", "tools", name], "name": name}


# ---------------------------------------------------------------------------
# Task 1 tests — chain / llm / tool spans
# ---------------------------------------------------------------------------


@pytest.mark.req("SDK-09")
def test_chain_start_end_creates_agent_span():
    """Handler creates an agent-typed span for chain start/end."""
    handler = MorseCallbackHandler("test-agent")
    run_id = uuid.uuid4()

    with tracing.run("test-agent"):
        handler.on_chain_start(
            _chain_serialized("TestChain"), {"question": "hi"}, run_id=run_id
        )
        handler.on_chain_end({"answer": "hello"}, run_id=run_id)

    trace = tracing._get_last_trace()
    assert trace is not None
    spans = trace["spans"]
    # Find the chain span (type=agent, name=TestChain)
    agent_spans = [
        s for s in spans if s.get("type") == "agent" and s.get("name") == "TestChain"
    ]
    assert len(agent_spans) == 1
    span = agent_spans[0]
    assert span["status"] == "completed"
    assert span["input_data"] == {"question": "hi"}
    assert span["output_data"] == {"answer": "hello"}


@pytest.mark.req("SDK-09")
def test_llm_span_captures_tokens():
    """Handler creates an llm span with model name and token usage."""
    handler = MorseCallbackHandler("test-agent")
    run_id = uuid.uuid4()
    llm_result = MockLLMResult(
        text="The answer is 42",
        tokens={
            "token_usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
            }
        },
    )

    with tracing.run("test-agent"):
        handler.on_llm_start(
            _llm_serialized("FakeLLM"),
            ["What is the answer?"],
            run_id=run_id,
            invocation_params={"model_name": "gpt-4o"},
        )
        handler.on_llm_end(llm_result, run_id=run_id)

    trace = tracing._get_last_trace()
    assert trace is not None
    llm_spans = [s for s in trace["spans"] if s.get("type") == "llm"]
    assert len(llm_spans) == 1
    s = llm_spans[0]
    assert s["model"] == "gpt-4o"
    assert s["input_tokens"] == 10
    assert s["output_tokens"] == 20
    assert s["total_tokens"] == 30
    assert s["status"] == "completed"


@pytest.mark.req("SDK-09")
def test_handler_restores_parent_trace_after_contextvar_loss():
    """Callbacks attach to an explicit parent run even after ContextVar loss."""
    run_id = uuid.uuid4()
    llm_result = MockLLMResult(
        text="detached response",
        tokens={
            "token_usage": {
                "prompt_tokens": 7,
                "completion_tokens": 11,
                "total_tokens": 18,
            }
        },
    )

    with tracing.run("parent-agent"):
        handler = MorseCallbackHandler("handler-agent")
        token = tracing._active_trace.set(None)
        try:
            handler.on_llm_start(
                _llm_serialized("DetachedLLM"),
                ["prompt"],
                run_id=run_id,
                invocation_params={"model_name": "gpt-4o-mini"},
            )
            handler.on_llm_end(llm_result, run_id=run_id)
        finally:
            tracing._active_trace.reset(token)

    trace = tracing._get_last_trace()
    assert trace is not None
    assert trace["trace"]["agent"] == "parent-agent"
    assert handler._run_context is None

    llm_spans = [s for s in trace["spans"] if s.get("type") == "llm"]
    assert len(llm_spans) == 1
    assert llm_spans[0]["name"] == "DetachedLLM"
    assert llm_spans[0]["total_tokens"] == 18


@pytest.mark.req("SDK-09")
def test_handler_reuse_after_parent_trace_close_starts_new_trace():
    """A stale captured parent state is not reused after its span stack closes."""
    with tracing.run("parent-agent"):
        handler = MorseCallbackHandler("handler-agent")

    tracing._last_trace = None
    run_id = uuid.uuid4()

    handler.on_chain_start(
        _chain_serialized("FreshAutoChain"),
        {"question": "new run"},
        run_id=run_id,
    )
    handler.on_chain_end({"answer": "new trace"}, run_id=run_id)

    trace = tracing._get_last_trace()
    assert trace is not None
    assert trace["trace"]["agent"] == "handler-agent"
    assert [s["name"] for s in trace["spans"] if s.get("type") == "agent"] == [
        "handler-agent",
        "FreshAutoChain",
    ]


@pytest.mark.req("SDK-09")
def test_tool_span_captures_io():
    """Handler creates a tool span with input and output data."""
    handler = MorseCallbackHandler("test-agent")
    run_id = uuid.uuid4()

    with tracing.run("test-agent"):
        handler.on_tool_start(_tool_serialized("calculator"), "2+2", run_id=run_id)
        handler.on_tool_end("4", run_id=run_id)

    trace = tracing._get_last_trace()
    assert trace is not None
    tool_spans = [s for s in trace["spans"] if s.get("type") == "tool"]
    assert len(tool_spans) == 1
    s = tool_spans[0]
    assert s.get("input_data") == {"input": "2+2"}
    assert s.get("output_data") == {"output": "4"}
    assert s["status"] == "completed"


@pytest.mark.req("SDK-09")
def test_error_sets_failed_status():
    """on_chain_error closes span with failed status and error_message."""
    handler = MorseCallbackHandler("test-agent")
    run_id = uuid.uuid4()
    error = ValueError("chain blew up")

    with tracing.run("test-agent"):
        handler.on_chain_start(_chain_serialized("BrokenChain"), {}, run_id=run_id)
        handler.on_chain_error(error, run_id=run_id)

    trace = tracing._get_last_trace()
    assert trace is not None
    failed_spans = [s for s in trace["spans"] if s.get("name") == "BrokenChain"]
    assert len(failed_spans) == 1
    s = failed_spans[0]
    assert s["status"] == "failed"
    assert "chain blew up" in s.get("error_message", "")


@pytest.mark.req("SDK-09")
def test_tool_error_sets_failed_status():
    """on_tool_error closes span with failed status."""
    handler = MorseCallbackHandler("test-agent")
    run_id = uuid.uuid4()
    error = RuntimeError("tool crashed")

    with tracing.run("test-agent"):
        handler.on_tool_start(_tool_serialized("broken_tool"), "input", run_id=run_id)
        handler.on_tool_error(error, run_id=run_id)

    trace = tracing._get_last_trace()
    assert trace is not None
    tool_spans = [s for s in trace["spans"] if s.get("type") == "tool"]
    assert len(tool_spans) == 1
    assert tool_spans[0]["status"] == "failed"
    assert "tool crashed" in tool_spans[0].get("error_message", "")


@pytest.mark.req("SDK-09")
def test_llm_error_sets_failed_status():
    """on_llm_error closes span with failed status."""
    handler = MorseCallbackHandler("test-agent")
    run_id = uuid.uuid4()
    error = ConnectionError("LLM unavailable")

    with tracing.run("test-agent"):
        handler.on_llm_start(_llm_serialized("FakeLLM"), ["prompt"], run_id=run_id)
        handler.on_llm_error(error, run_id=run_id)

    trace = tracing._get_last_trace()
    assert trace is not None
    llm_spans = [s for s in trace["spans"] if s.get("type") == "llm"]
    assert len(llm_spans) == 1
    assert llm_spans[0]["status"] == "failed"


@pytest.mark.req("SDK-09")
def test_auto_trace_creation():
    """Handler auto-creates a trace context when none exists."""
    # Reset last trace so we can tell this one is new
    tracing._last_trace = None

    handler = MorseCallbackHandler("auto-agent")
    run_id = uuid.uuid4()

    # No enclosing run() — handler must auto-create
    handler.on_chain_start(
        _chain_serialized("AutoChain"), {"q": "auto?"}, run_id=run_id
    )
    handler.on_chain_end({"a": "yes"}, run_id=run_id)

    trace = tracing._get_last_trace()
    assert trace is not None
    assert trace["trace"]["agent"] == "auto-agent"
    agent_spans = [s for s in trace["spans"] if s.get("name") == "AutoChain"]
    assert len(agent_spans) == 1


@pytest.mark.req("SDK-09")
def test_nested_spans_have_correct_parent():
    """chain_start → llm_start → llm_end → tool_start → tool_end → chain_end produces correct hierarchy."""
    handler = MorseCallbackHandler("test-agent")
    chain_run_id = uuid.uuid4()
    llm_run_id = uuid.uuid4()
    tool_run_id = uuid.uuid4()

    with tracing.run("test-agent"):
        handler.on_chain_start(_chain_serialized("OuterChain"), {}, run_id=chain_run_id)
        handler.on_llm_start(
            _llm_serialized("FakeLLM"),
            ["prompt"],
            run_id=llm_run_id,
            parent_run_id=chain_run_id,
        )
        handler.on_llm_end(MockLLMResult(), run_id=llm_run_id)
        handler.on_tool_start(
            _tool_serialized("mytool"),
            "in",
            run_id=tool_run_id,
            parent_run_id=chain_run_id,
        )
        handler.on_tool_end("out", run_id=tool_run_id)
        handler.on_chain_end({}, run_id=chain_run_id)

    trace = tracing._get_last_trace()
    assert trace is not None
    spans = trace["spans"]

    by_name = {s["name"]: s for s in spans}
    assert "OuterChain" in by_name
    assert "FakeLLM" in by_name
    assert "mytool" in by_name

    chain_span_id = by_name["OuterChain"]["span_id"]
    # LLM and tool spans should have the chain span as parent
    assert by_name["FakeLLM"].get("parent_span_id") == chain_span_id
    assert by_name["mytool"].get("parent_span_id") == chain_span_id


@pytest.mark.req("SDK-09")
def test_graceful_fallback_when_no_langchain():
    """MorseCallbackHandler can be imported and instantiated without langchain."""
    # If we got here without ImportError, the import succeeded
    h = MorseCallbackHandler("fallback-test")
    assert h is not None
    assert h.agent_name == "fallback-test"


@pytest.mark.req("SDK-09")
def test_callbacks_tolerate_none_serialized():
    """LangChain passes ``serialized=None`` for some Runnable graphs in newer
    langchain_core releases. The handler must NOT raise into the host app —
    silent no-op is the SDK contract.

    Regression: dogfood Copilot LangGraph chain triggered
    ``AttributeError("'NoneType' object has no attribute 'get'")`` from
    ``on_chain_start`` because the previous code did ``serialized.get(...)``
    unconditionally.
    """
    handler = MorseCallbackHandler("none-serialized-test")
    run_id_chain = uuid.uuid4()
    run_id_llm = uuid.uuid4()
    run_id_tool = uuid.uuid4()

    with tracing.run("none-serialized-test"):
        # All three on_*_start variants must accept serialized=None.
        handler.on_chain_start(None, {"q": "hi"}, run_id=run_id_chain)
        handler.on_chain_end({"a": "ok"}, run_id=run_id_chain)

        handler.on_llm_start(None, ["prompt"], run_id=run_id_llm)
        handler.on_llm_end(MockLLMResult("done"), run_id=run_id_llm)

        handler.on_tool_start(None, "input", run_id=run_id_tool)
        handler.on_tool_end("output", run_id=run_id_tool)

    # No exception means pass; spans should have used the "chain"/"llm"/"tool"
    # default names since serialized was None.
    trace = tracing._get_last_trace()
    assert trace is not None
    names = {s.get("name") for s in trace["spans"]}
    assert "chain" in names
    assert "llm" in names
    assert "tool" in names


# ---------------------------------------------------------------------------
# AGE-240: usage + cost extraction across provider shapes
# ---------------------------------------------------------------------------


def _anthropic_lc_v3_result(
    text: str = "hi", model: str = "claude-sonnet-4-5"
) -> MockLLMResult:
    """Mimic LangChain v0.3 Anthropic LLMResult: usage on message.usage_metadata."""

    class _Msg:
        def __init__(self) -> None:
            self.usage_metadata = {
                "input_tokens": 100,
                "output_tokens": 50,
                "total_tokens": 150,
                "input_token_details": {"cache_read": 0},
            }
            self.response_metadata = {"model_name": model}

    class _Gen:
        def __init__(self) -> None:
            self.text = text
            self.message = _Msg()

    r = MockLLMResult(text=text)
    r.generations = [[_Gen()]]
    return r


@pytest.mark.req("SDK-09")
def test_on_llm_end_extracts_anthropic_usage_and_cost():
    """LC v0.3 Anthropic shape: usage on message.usage_metadata + model on response_metadata.

    Regression: dogfood Copilot ChatAnthropic spans had model set but
    tokens/cost null because the previous code only read OpenAI's
    ``llm_output.token_usage`` shape.
    """
    handler = MorseCallbackHandler("anthropic-usage-test")
    run_id = uuid.uuid4()
    result = _anthropic_lc_v3_result(model="claude-sonnet-4-5-20250929")

    with tracing.run("anthropic-usage-test"):
        handler.on_llm_start({"id": ["ChatAnthropic"]}, ["q"], run_id=run_id)
        handler.on_llm_end(result, run_id=run_id)

    trace = tracing._get_last_trace()
    assert trace is not None
    llm_spans = [s for s in trace["spans"] if s.get("type") == "llm"]
    assert len(llm_spans) == 1
    s = llm_spans[0]
    assert s["input_tokens"] == 100
    assert s["output_tokens"] == 50
    assert s["total_tokens"] == 150
    # Cost should be populated via the pricing table for a known model.
    assert s.get("cost_usd") is not None and s["cost_usd"] > 0


@pytest.mark.req("SDK-09")
def test_on_llm_end_extracts_anthropic_legacy_usage():
    """Older Anthropic LangChain integration: usage in llm_output['usage']."""
    handler = MorseCallbackHandler("anthropic-legacy-test")
    run_id = uuid.uuid4()
    result = MockLLMResult(
        text="hi", tokens={"usage": {"input_tokens": 10, "output_tokens": 20}}
    )

    with tracing.run("anthropic-legacy-test"):
        handler.on_llm_start({"id": ["ChatAnthropic"]}, ["q"], run_id=run_id)
        handler.on_llm_end(result, run_id=run_id)

    trace = tracing._get_last_trace()
    llm_spans = [s for s in trace["spans"] if s.get("type") == "llm"]
    assert len(llm_spans) == 1
    assert llm_spans[0]["input_tokens"] == 10
    assert llm_spans[0]["output_tokens"] == 20


@pytest.mark.req("SDK-09")
def test_on_llm_end_extracts_response_metadata_token_usage_and_chat_content():
    """Some ChatOpenAI-compatible integrations put usage under response_metadata."""
    handler = MorseCallbackHandler("openai-response-metadata-test")
    run_id = uuid.uuid4()

    class _Msg:
        usage_metadata = {}
        response_metadata = {
            "model_name": "gpt-4o-mini-2024-07-18",
            "token_usage": {
                "prompt_tokens": 12,
                "completion_tokens": 7,
                "total_tokens": 19,
            },
        }
        content = "chat content fallback"

    class _Gen:
        text = ""
        message = _Msg()

    result = MockLLMResult(text="")
    result.generations = [[_Gen()]]

    with tracing.run("openai-response-metadata-test"):
        handler.on_llm_start({"id": ["ChatOpenAI"]}, ["q"], run_id=run_id)
        handler.on_llm_end(result, run_id=run_id)

    trace = tracing._get_last_trace()
    llm_spans = [s for s in trace["spans"] if s.get("type") == "llm"]
    assert len(llm_spans) == 1
    assert llm_spans[0]["input_tokens"] == 12
    assert llm_spans[0]["output_tokens"] == 7
    assert llm_spans[0]["total_tokens"] == 19
    assert llm_spans[0]["model"] == "gpt-4o-mini-2024-07-18"
    assert llm_spans[0]["output_data"] == {"text": "chat content fallback"}


@pytest.mark.req("SDK-09")
def test_on_llm_end_still_handles_openai_legacy_shape():
    """Backwards compatibility: OpenAI ``llm_output['token_usage']`` path still works."""
    handler = MorseCallbackHandler("openai-legacy-test")
    run_id = uuid.uuid4()
    result = MockLLMResult(
        text="hi",
        tokens={
            "token_usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
            }
        },
    )

    with tracing.run("openai-legacy-test"):
        handler.on_llm_start({"id": ["OpenAI"]}, ["q"], run_id=run_id)
        handler.on_llm_end(result, run_id=run_id)

    trace = tracing._get_last_trace()
    llm_spans = [s for s in trace["spans"] if s.get("type") == "llm"]
    assert llm_spans[0]["input_tokens"] == 10
    assert llm_spans[0]["output_tokens"] == 20
    assert llm_spans[0]["total_tokens"] == 30


@pytest.mark.req("MHQ-1022")
def test_langgraph_thread_id_from_run_metadata_lands_on_the_trace():
    """MHQ-1022: a checkpointed LangGraph conversation must group by thread.

    ``configurable.thread_id`` reaches a callback handler only via the run
    metadata, so this is the one signal available before the trace is emitted.
    """
    handler = MorseCallbackHandler("chat-agent")
    run_id = uuid.uuid4()

    handler.on_chain_start(
        _chain_serialized("LangGraph"),
        {"messages": []},
        run_id=run_id,
        metadata={"thread_id": "conv-77", "langgraph_step": 1},
    )
    handler.on_chain_end({"messages": []}, run_id=run_id)

    assert tracing._get_last_trace()["trace"]["thread_id"] == "conv-77"
