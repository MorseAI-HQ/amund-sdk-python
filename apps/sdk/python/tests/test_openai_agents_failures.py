"""MHQ-1406 — OpenAI Agents adapter must record failures and the served model.

Adapter: `apps/sdk/python/src/morse_ai/adapters/openai_agents.py`

Real `Runner` + fake `Model` objects, in-process, no network. Three defects
from the 2026-10-18 dogfood round:

1. a model-call exception under a *bare* ``Runner`` left the trace
   ``completed`` (only the explicit ``morse_ai.run()`` path recorded it);
2. a function tool that raised was swallowed by the Agents SDK (fed to the
   model as an error string) and showed as a successful tool span;
3. an ``Agent`` with no ``model=`` produced ``model=NULL`` and so no cost.
"""

# ruff: noqa: E501
from __future__ import annotations

import asyncio
import uuid
from types import SimpleNamespace

import pytest

pytest.importorskip("agents", reason="openai-agents not installed")

from agents import (  # noqa: E402
    Agent,
    Model,
    ModelResponse,
    RunConfig,
    Runner,
    Usage,
    function_tool,
    set_tracing_disabled,
)
from openai.types.responses import (  # noqa: E402
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
)

import morse_ai  # noqa: E402
from morse_ai import tracing  # noqa: E402
from morse_ai.adapters import openai_agents as adapter  # noqa: E402
from morse_ai.spans import SpanData, SpanType  # noqa: E402
from morse_ai.tracing import _active_trace, _TraceState  # noqa: E402

set_tracing_disabled(True)


@pytest.fixture(autouse=True)
def _installed():
    morse_ai._reset_for_testing()
    adapter.uninstall()
    adapter.install()
    yield
    # _bare_state() sets the active-trace ContextVar directly; a leftover state
    # made later tests (e.g. test_langchain_adapter::test_auto_trace_creation)
    # see an already-open trace.
    _active_trace.set(None)
    morse_ai._reset_for_testing()
    adapter.uninstall()


def _text(text: str) -> ResponseOutputMessage:
    return ResponseOutputMessage(
        id="m1",
        type="message",
        role="assistant",
        status="completed",
        content=[ResponseOutputText(type="output_text", text=text, annotations=[])],
    )


def _usage() -> Usage:
    return Usage(requests=1, input_tokens=20, output_tokens=13, total_tokens=33)


class _Boom(Exception):
    pass


class _FailingModel(Model):
    async def get_response(self, *a, **k):
        raise _Boom("model not found")

    def stream_response(self, *a, **k):
        async def _gen():
            raise _Boom("model not found")
            yield  # pragma: no cover

        return _gen()


class _ScriptedModel(Model):
    def __init__(self, outputs):
        self._outputs = list(outputs)

    async def get_response(self, *a, **k):
        return ModelResponse(
            output=[self._outputs.pop(0)], usage=_usage(), response_id=None
        )

    def stream_response(self, *a, **k):
        raise NotImplementedError


def _last():
    t = tracing._get_last_trace()
    assert t is not None
    return t


def _root(t):
    roots = [s for s in t["spans"] if not s.get("parent_span_id")]
    assert len(roots) == 1
    return roots[0]


def test_bare_run_sync_model_error_marks_trace_failed():
    agent = Agent(name="bare-fail", model=_FailingModel())
    with pytest.raises(_Boom):
        Runner.run_sync(agent, "hi")
    t = _last()
    root = _root(t)
    assert root["status"] == "failed"
    assert "model not found" in (root.get("error_message") or "")
    assert root["exception_type"].endswith("_Boom")
    assert t["trace"]["status"] == "failed"


def test_bare_async_run_model_error_marks_trace_failed():
    agent = Agent(name="bare-fail-async", model=_FailingModel())

    async def main():
        await Runner.run(agent, "hi")

    with pytest.raises(_Boom):
        asyncio.run(main())
    t = _last()
    assert _root(t)["status"] == "failed"
    assert t["trace"]["status"] == "failed"


def test_bare_streamed_run_model_error_marks_trace_failed():
    agent = Agent(name="bare-fail-stream", model=_FailingModel())

    async def main():
        result = Runner.run_streamed(agent, "hi")
        async for _ in result.stream_events():
            pass

    with pytest.raises(_Boom):
        asyncio.run(main())
    t = _last()
    assert _root(t)["status"] == "failed"
    assert t["trace"]["status"] == "failed"


def test_successful_bare_run_stays_completed():
    agent = Agent(name="ok", model=_ScriptedModel([_text("hello")]))
    Runner.run_sync(agent, "hi")
    t = _last()
    assert _root(t)["status"] == "completed"
    assert t["trace"]["status"] == "completed"


@function_tool
def seat_lookup(row: int) -> str:
    """Look up a seat."""
    raise RuntimeError("seat service exploded")


@function_tool
def ok_tool(a: int) -> str:
    """Fine tool."""
    return "fine"


def _call(name: str, args: str) -> ResponseFunctionToolCall:
    return ResponseFunctionToolCall(
        type="function_call", call_id="c1", name=name, arguments=args, id="fc1"
    )


def test_tool_exception_swallowed_by_sdk_is_recorded_on_tool_span():
    agent = Agent(
        name="tool-boom",
        model=_ScriptedModel([_call("seat_lookup", '{"row": 3}'), _text("sorry")]),
        tools=[seat_lookup],
    )
    Runner.run_sync(agent, "seat?")  # the SDK feeds the error to the model
    t = _last()
    tools = [s for s in t["spans"] if s["type"] == "tool"]
    assert len(tools) == 1
    assert tools[0]["status"] == "failed"
    assert "seat service exploded" in tools[0]["error_message"]
    assert tools[0]["exception_type"] == "RuntimeError"
    # The agent run itself recovered — only the tool span is failed.
    assert _root(t)["status"] == "completed"


def test_successful_tool_stays_completed():
    agent = Agent(
        name="tool-ok",
        model=_ScriptedModel([_call("ok_tool", '{"a": 1}'), _text("done")]),
        tools=[ok_tool],
    )
    Runner.run_sync(agent, "go")
    tools = [s for s in _last()["spans"] if s["type"] == "tool"]
    assert tools[0]["status"] == "completed"
    assert not tools[0].get("error_message")


def test_model_resolved_when_agent_model_is_unset(monkeypatch):
    """Quickstart shape: no `model=` on the Agent. The SDK runs its default
    model, so the span must carry that name or the server cannot price it."""
    monkeypatch.setenv("OPENAI_DEFAULT_MODEL", "gpt-4o-mini")
    agent = Agent(name="no-model")  # model=None
    assert agent.model is None
    cfg = RunConfig(model=_ScriptedModel([_text("hi")]))
    Runner.run_sync(agent, "hi", run_config=cfg)
    llm = [s for s in _last()["spans"] if s["type"] == "llm"][0]
    # run_config.model is a nameless Model object -> SDK default name
    assert llm["model"] == "gpt-4o-mini"
    assert llm["cost_usd"] is not None


def _bare_state():
    state = _TraceState(trace_id=uuid.uuid4().hex, agent_name="x")
    root = SpanData(trace_id=state.trace_id, type=SpanType.AGENT, name="x")
    state.span_stack.append(root)
    _active_trace.set(state)
    return state, root


def test_run_config_model_string_beats_agent_model():
    """`RunConfig.model` overrides `Agent.model` in the SDK, so it must win."""
    agent = Agent(name="cfg", model="gpt-4o")
    state, root = _bare_state()
    HooksCls = adapter._build_av_run_hooks()
    hooks = HooksCls(
        state,
        root,
        {"name": "cfg"},
        None,
        run_config=SimpleNamespace(model="gpt-4o-mini"),
    )
    asyncio.run(hooks.on_llm_start(SimpleNamespace(), agent, None, []))
    resp = SimpleNamespace(
        usage=SimpleNamespace(
            input_tokens=10, output_tokens=5, total_tokens=15, input_tokens_details=None
        ),
        output=[],
    )
    asyncio.run(hooks.on_llm_end(SimpleNamespace(), agent, resp))
    llm = [s for s in state.completed_spans if s.type == SpanType.LLM][0]
    assert llm.model == "gpt-4o-mini"
    assert llm.cost_usd is not None
