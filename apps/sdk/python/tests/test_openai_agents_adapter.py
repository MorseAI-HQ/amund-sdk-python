"""Tests for the OpenAI Agents (`openai-agents`) adapter.

Adapter file: `apps/sdk/python/src/morse_ai/adapters/openai_agents.py`

These tests exercise the adapter at the lifecycle-hook level. We don't make
real LLM calls — instead we drive the adapter's `_MorseRunHooks` instance
directly with synthesized objects shaped like the real `agents` SDK types.
This is the same testing strategy used by the Anthropic adapter tests and
keeps the suite hermetic.
"""

# ruff: noqa: E501, SIM117
from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace
from typing import Any

import pytest

import morse_ai
from morse_ai.adapters import openai_agents as adapter
from morse_ai.spans import SpanData, SpanType
from morse_ai.tracing import (
    _active_trace,
    _TraceState,
)

pytestmark = pytest.mark.skipif(
    "agents" not in sys.modules
    and pytest.importorskip("agents", reason="openai-agents not installed") is None,
    reason="openai-agents package required",
)


@pytest.fixture(autouse=True)
def _reset_singleton():
    morse_ai._reset_for_testing()
    adapter.uninstall()
    adapter._session_ids.clear()
    yield
    morse_ai._reset_for_testing()
    adapter.uninstall()
    adapter._session_ids.clear()


# ---------------------------------------------------------------------------
# Helpers — simulate openai-agents types without running a real LLM.
# ---------------------------------------------------------------------------


def _start_trace(agent_name: str = "test-agent") -> tuple[_TraceState, SpanData]:
    """Create a fake active trace and root agent span."""
    import uuid

    state = _TraceState(trace_id=uuid.uuid4().hex, agent_name=agent_name)
    root_span = SpanData(
        trace_id=state.trace_id,
        type=SpanType.AGENT,
        name=agent_name,
    )
    state.span_stack.append(root_span)
    _active_trace.set(state)
    return state, root_span


def _make_usage(
    input_tokens: int = 100,
    output_tokens: int = 50,
    total_tokens: int | None = None,
    cached_tokens: int = 0,
) -> Any:
    return SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens or (input_tokens + output_tokens),
        input_tokens_details=SimpleNamespace(cached_tokens=cached_tokens),
    )


def _run_async(coro):
    """Run an async coroutine to completion in the current event loop, or a fresh one."""
    return (
        asyncio.get_event_loop().run_until_complete(coro)
        if False
        else asyncio.run(coro)
    )


# ---------------------------------------------------------------------------
# install / uninstall idempotency
# ---------------------------------------------------------------------------


def test_install_idempotent_when_agents_available():
    first = adapter.install()
    second = adapter.install()
    third = adapter.install()
    assert first is True
    assert second is True
    assert third is True
    assert adapter.is_installed() is True


def test_uninstall_restores_originals():
    from agents.run import Runner

    original_run = Runner.run
    original_run_sync = Runner.run_sync
    original_run_streamed = Runner.run_streamed

    adapter.install()
    assert Runner.run is not original_run

    adapter.uninstall()
    assert Runner.run is original_run
    assert Runner.run_sync is original_run_sync
    assert Runner.run_streamed is original_run_streamed
    assert adapter.is_installed() is False


def test_uninstall_idempotent_when_not_installed():
    adapter.uninstall()
    adapter.uninstall()  # double-call is fine
    assert adapter.is_installed() is False


# ---------------------------------------------------------------------------
# Lifecycle hooks — span emission
# ---------------------------------------------------------------------------


def test_on_llm_start_end_emits_llm_span_with_cost():
    state, root_span = _start_trace()
    holder = {"name": "agent_a"}
    HooksCls = adapter._build_av_run_hooks()
    hooks = HooksCls(state, root_span, holder, user_hooks=None)

    agent = SimpleNamespace(name="agent_a", model="gpt-4o")

    asyncio.run(
        hooks.on_llm_start(SimpleNamespace(), agent, system_prompt=None, input_items=[])
    )
    response = SimpleNamespace(usage=_make_usage(input_tokens=1000, output_tokens=500))
    asyncio.run(hooks.on_llm_end(SimpleNamespace(), agent, response))

    llm_spans = [s for s in state.completed_spans if s.type == SpanType.LLM]
    assert len(llm_spans) == 1
    span = llm_spans[0]
    assert span.model == "gpt-4o"
    assert span.input_tokens == 1000
    assert span.output_tokens == 500
    assert span.total_tokens == 1500
    # gpt-4o: $2.5/Mtok input, $10/Mtok output
    expected_cost = round((1000 / 1_000_000) * 2.5 + (500 / 1_000_000) * 10.0, 6)
    assert span.cost_usd == expected_cost
    # Parent linkage
    assert span.parent_span_id == root_span.span_id


def test_cost_within_5pct_of_openai_billing():
    """Cost matches OpenAI gpt-4o pricing within ±5%."""
    state, root_span = _start_trace()
    holder = {"name": "agent_a"}
    HooksCls = adapter._build_av_run_hooks()
    hooks = HooksCls(state, root_span, holder, user_hooks=None)

    agent = SimpleNamespace(name="agent_a", model="gpt-4o")

    asyncio.run(
        hooks.on_llm_start(SimpleNamespace(), agent, system_prompt=None, input_items=[])
    )
    response = SimpleNamespace(
        usage=_make_usage(input_tokens=10_000, output_tokens=2_000)
    )
    asyncio.run(hooks.on_llm_end(SimpleNamespace(), agent, response))

    span = state.completed_spans[-1]
    # Authoritative: $2.5/MTok input, $10/MTok output
    expected = (10_000 / 1e6) * 2.5 + (2_000 / 1e6) * 10.0
    assert abs(span.cost_usd - expected) / expected < 0.05


def test_function_tool_emits_tool_span():
    state, root_span = _start_trace()
    holder = {"name": "agent_a"}
    HooksCls = adapter._build_av_run_hooks()
    hooks = HooksCls(state, root_span, holder, user_hooks=None)

    agent = SimpleNamespace(name="agent_a", model="gpt-4o")
    tool = SimpleNamespace(name="lookup_user")
    context = SimpleNamespace(tool_arguments={"user_id": "u1"})

    asyncio.run(hooks.on_tool_start(context, agent, tool))
    asyncio.run(hooks.on_tool_end(context, agent, tool, result={"ok": True}))

    tool_spans = [s for s in state.completed_spans if s.type == SpanType.TOOL]
    assert len(tool_spans) == 1
    span = tool_spans[0]
    assert span.name == "lookup_user"
    assert span.parent_span_id == root_span.span_id
    assert span.input_data == {"user_id": "u1"}
    assert span.output_data is not None
    assert "ok" in span.output_data


def test_handoff_emits_handoff_span_and_subsequent_llm_uses_target_name():
    state, root_span = _start_trace()
    holder = {"name": "agent_a"}
    HooksCls = adapter._build_av_run_hooks()
    hooks = HooksCls(state, root_span, holder, user_hooks=None)

    agent_a = SimpleNamespace(name="agent_a", model="gpt-4o")
    agent_b = SimpleNamespace(name="agent_b", model="gpt-4o")

    asyncio.run(hooks.on_handoff(SimpleNamespace(), agent_a, agent_b))

    # MHQ-1025: a navigable span, not an array on the root span.
    handoff_spans = [s for s in state.completed_spans if s.type == SpanType.HANDOFF]
    assert len(handoff_spans) == 1
    hs = handoff_spans[0]
    assert hs.metadata == {"from_agent": "agent_a", "to_agent": "agent_b"}
    assert hs.parent_span_id == root_span.span_id
    assert hs.ended_at is not None and hs.duration_ms is not None
    assert (root_span.metadata or {}).get("handoffs") is None
    # Subsequent LLM start should pick up agent_b as the current agent
    assert holder["name"] == "agent_b"

    asyncio.run(
        hooks.on_llm_start(
            SimpleNamespace(), agent_b, system_prompt=None, input_items=[]
        )
    )
    response = SimpleNamespace(usage=_make_usage())
    asyncio.run(hooks.on_llm_end(SimpleNamespace(), agent_b, response))

    llm_span = state.completed_spans[-1]
    assert llm_span.metadata is not None
    assert llm_span.metadata.get("agent_name") == "agent_b"


def test_handoff_emits_child_agent_span_and_reparents_subsequent_spans():
    """A handoff target becomes a first-class child AGENT span (a tree node),
    and LLM/tool spans after the handoff nest under it, not the root."""
    state, root_span = _start_trace("triage")
    holder = {"name": "triage"}
    HooksCls = adapter._build_av_run_hooks()
    hooks = HooksCls(state, root_span, holder, user_hooks=None)

    triage = SimpleNamespace(name="triage", model="gpt-4o")
    booking = SimpleNamespace(name="booking", model="gpt-4o")

    asyncio.run(hooks.on_handoff(SimpleNamespace(), triage, booking))

    # The handoff target is now a distinct AGENT span, child of the root.
    agent_spans = [s for s in state.completed_spans if s.type == SpanType.AGENT]
    assert len(agent_spans) == 1
    booking_span = agent_spans[0]
    assert booking_span.name == "booking"
    assert booking_span.parent_span_id == root_span.span_id
    assert booking_span.metadata.get("askl_handoff_from") == "triage"

    # An LLM call after the handoff nests under the booking agent, not root.
    asyncio.run(
        hooks.on_llm_start(
            SimpleNamespace(), booking, system_prompt=None, input_items=[]
        )
    )
    asyncio.run(
        hooks.on_llm_end(
            SimpleNamespace(), booking, SimpleNamespace(usage=_make_usage())
        )
    )
    llm_span = next(s for s in state.completed_spans if s.type == SpanType.LLM)
    assert llm_span.parent_span_id == booking_span.span_id

    # At run end the open agent span gets finalized (duration set).
    hooks.finalize_open_agents()
    assert booking_span.ended_at is not None
    assert booking_span.duration_ms is not None


def test_two_hop_handoff_chain_is_navigable():
    """MHQ-1025: each hop is its own span, parented to the agent that made the
    decision — so triage→booking→refunds reads as a chain, not a flat list."""
    state, root_span = _start_trace("triage")
    HooksCls = adapter._build_av_run_hooks()
    hooks = HooksCls(state, root_span, {"name": "triage"}, user_hooks=None)

    triage = SimpleNamespace(name="triage", model="gpt-4o")
    booking = SimpleNamespace(name="booking", model="gpt-4o")
    refunds = SimpleNamespace(name="refunds", model="gpt-4o")

    asyncio.run(hooks.on_handoff(SimpleNamespace(), triage, booking))
    asyncio.run(hooks.on_handoff(SimpleNamespace(), booking, refunds))

    handoffs = [s for s in state.completed_spans if s.type == SpanType.HANDOFF]
    agents = {s.name: s for s in state.completed_spans if s.type == SpanType.AGENT}
    assert [h.name for h in handoffs] == ["triage → booking", "booking → refunds"]
    # Hop 1 is made by the root (triage); hop 2 by the booking agent span.
    assert handoffs[0].parent_span_id == root_span.span_id
    assert handoffs[1].parent_span_id == agents["booking"].span_id


def test_session_id_resolution_consistent_across_calls():
    session = SimpleNamespace()  # no session_id attribute
    sid1 = adapter._resolve_session_id(session)
    sid2 = adapter._resolve_session_id(session)
    assert sid1 is not None
    assert sid1 == sid2

    session2 = SimpleNamespace()
    sid3 = adapter._resolve_session_id(session2)
    assert sid3 != sid1


def test_attach_run_metadata_sets_session_and_structured_output_type():
    root_span = SpanData(
        trace_id="t1",
        type=SpanType.AGENT,
        name="agent_a",
    )

    class MyOutput:
        pass

    starting = SimpleNamespace(name="agent_a", output_type=MyOutput)
    session = SimpleNamespace()

    adapter._attach_run_metadata(root_span, starting, None, session)

    assert root_span.metadata is not None
    assert root_span.metadata["agent_name"] == "agent_a"
    assert root_span.metadata["session_id"].startswith("av-session-")
    assert root_span.metadata["structured_output_type"] == "MyOutput"


def test_nested_run_does_not_overwrite_root_attribution():
    """MHQ-1023: a guardrail's nested Runner.run reuses the outer root span."""
    root_span = SpanData(trace_id="t1", type=SpanType.AGENT, name="FAQ Agent")

    class AgentOut:
        pass

    class GuardrailOut:
        pass

    adapter._attach_run_metadata(
        root_span,
        SimpleNamespace(name="FAQ Agent", output_type=AgentOut),
        {"q": "real input"},
        None,
    )
    adapter._attach_run_metadata(
        root_span,
        SimpleNamespace(name="Jailbreak Guardrail", output_type=GuardrailOut),
        {"q": "guardrail input"},
        None,
    )
    adapter._finalize_root_with_output(
        root_span,
        SimpleNamespace(output_type=AgentOut),
        SimpleNamespace(final_output="agent answer"),
    )
    adapter._finalize_root_with_output(
        root_span,
        SimpleNamespace(output_type=GuardrailOut),
        SimpleNamespace(final_output="guardrail verdict"),
    )

    assert root_span.metadata["agent_name"] == root_span.name == "FAQ Agent"
    assert root_span.metadata["structured_output_type"] == "AgentOut"
    assert root_span.metadata["structured_output"] == {"text": "agent answer"}
    assert root_span.input_data == {"q": "real input"}


def test_finalize_root_with_output_attaches_structured_output():
    root_span = SpanData(trace_id="t1", type=SpanType.AGENT, name="agent_a")

    class MyOutput:
        pass

    starting = SimpleNamespace(name="agent_a", output_type=MyOutput)
    result = SimpleNamespace(final_output={"hypothesis": "x", "confidence": "high"})

    adapter._finalize_root_with_output(root_span, starting, result)

    assert root_span.metadata is not None
    assert "structured_output" in root_span.metadata
    assert "hypothesis" in root_span.metadata["structured_output"]


# ---------------------------------------------------------------------------
# User-hooks composition
# ---------------------------------------------------------------------------


def test_user_hooks_are_invoked_alongside_av_hooks():
    state, root_span = _start_trace()
    holder = {"name": "agent_a"}

    user_calls = []

    class UserHooks:
        async def on_llm_start(self, *args, **kwargs):
            user_calls.append("llm_start")

        async def on_llm_end(self, *args, **kwargs):
            user_calls.append("llm_end")

        async def on_tool_start(self, *args, **kwargs):
            user_calls.append("tool_start")

        async def on_tool_end(self, *args, **kwargs):
            user_calls.append("tool_end")

        async def on_handoff(self, *args, **kwargs):
            user_calls.append("handoff")

        async def on_agent_start(self, *args, **kwargs):
            user_calls.append("agent_start")

        async def on_agent_end(self, *args, **kwargs):
            user_calls.append("agent_end")

    HooksCls = adapter._build_av_run_hooks()
    hooks = HooksCls(state, root_span, holder, user_hooks=UserHooks())

    agent = SimpleNamespace(name="agent_a", model="gpt-4o")
    tool = SimpleNamespace(name="t")
    ctx = SimpleNamespace(tool_input={})

    asyncio.run(hooks.on_agent_start(ctx, agent))
    asyncio.run(hooks.on_llm_start(ctx, agent, None, []))
    asyncio.run(hooks.on_llm_end(ctx, agent, SimpleNamespace(usage=_make_usage())))
    asyncio.run(hooks.on_tool_start(ctx, agent, tool))
    asyncio.run(hooks.on_tool_end(ctx, agent, tool, "ok"))
    asyncio.run(hooks.on_handoff(ctx, agent, agent))
    asyncio.run(hooks.on_agent_end(ctx, agent, "done"))

    assert user_calls == [
        "agent_start",
        "llm_start",
        "llm_end",
        "tool_start",
        "tool_end",
        "handoff",
        "agent_end",
    ]


def test_user_hook_exception_does_not_break_av_hooks():
    """Customer-supplied hooks raising must NOT break Morse's hooks."""
    state, root_span = _start_trace()
    holder = {"name": "agent_a"}

    class BadHooks:
        async def on_llm_start(self, *args, **kwargs):
            raise RuntimeError("boom")

        async def on_llm_end(self, *args, **kwargs):
            raise RuntimeError("boom")

    HooksCls = adapter._build_av_run_hooks()
    hooks = HooksCls(state, root_span, holder, user_hooks=BadHooks())

    agent = SimpleNamespace(name="agent_a", model="gpt-4o")
    asyncio.run(hooks.on_llm_start(SimpleNamespace(), agent, None, []))
    asyncio.run(
        hooks.on_llm_end(SimpleNamespace(), agent, SimpleNamespace(usage=_make_usage()))
    )

    # The LLM span must still be recorded despite user-hook failure.
    llm_spans = [s for s in state.completed_spans if s.type == SpanType.LLM]
    assert len(llm_spans) == 1


# ---------------------------------------------------------------------------
# auto_detect registration
# ---------------------------------------------------------------------------


def test_openai_agents_appears_in_adapter_targets_when_importable():
    from morse_ai._internal import auto_detect as ad

    targets = ad._adapter_targets()
    assert "agents" in targets
    assert targets["agents"] == "morse_ai.adapters.openai_agents"


def test_on_llm_end_prefers_response_model_over_agent_model():
    """With model routing, span.model + cost follow the model the API served,
    not the agent's configured model captured at on_llm_start."""
    state, root_span = _start_trace()
    holder = {"name": "agent_a"}
    HooksCls = adapter._build_av_run_hooks()
    hooks = HooksCls(state, root_span, holder, user_hooks=None)

    # Agent configured for gpt-4o, but the router served gpt-4o-mini.
    agent = SimpleNamespace(name="agent_a", model="gpt-4o")
    asyncio.run(
        hooks.on_llm_start(SimpleNamespace(), agent, system_prompt=None, input_items=[])
    )
    response = SimpleNamespace(
        model="gpt-4o-mini", usage=_make_usage(input_tokens=1000, output_tokens=500)
    )
    asyncio.run(hooks.on_llm_end(SimpleNamespace(), agent, response))

    span = [s for s in state.completed_spans if s.type == SpanType.LLM][0]
    assert span.model == "gpt-4o-mini"
    # priced at gpt-4o-mini ($0.15/$0.60 per MTok), not gpt-4o ($2.5/$10)
    expected_cost = round((1000 / 1_000_000) * 0.15 + (500 / 1_000_000) * 0.6, 6)
    assert span.cost_usd == expected_cost


# ---------------------------------------------------------------------------
# Guardrails (MHQ-1024)
# ---------------------------------------------------------------------------


def _input_guardrail(triggered: bool, reasoning: str = "looks fine"):
    """Build a real `InputGuardrail` whose check reports `triggered`."""
    from agents.guardrail import GuardrailFunctionOutput, InputGuardrail

    async def _check(context, agent, input):
        return GuardrailFunctionOutput(
            output_info={"reasoning": reasoning},
            tripwire_triggered=triggered,
        )

    return InputGuardrail(guardrail_function=_check, name="Relevance Guardrail")


def test_input_guardrail_emits_guardrail_span_when_passing():
    state, _root = _start_trace()
    adapter.install()
    guardrail = _input_guardrail(triggered=False)

    asyncio.run(
        guardrail.run(SimpleNamespace(name="triage"), "hello", SimpleNamespace())
    )

    spans = [s for s in state.completed_spans if s.type == SpanType.GUARDRAIL]
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "guardrail.Relevance Guardrail"
    assert span.status == "completed"
    assert span.metadata["guardrail_name"] == "Relevance Guardrail"
    assert span.metadata["guardrail_kind"] == "input"
    assert span.metadata["tripwire_triggered"] is False
    # The input that was checked, and the guardrail's own reasoning.
    assert span.input_data == {"text": "hello"}
    assert span.output_data == {"reasoning": "looks fine"}


def test_tripped_input_guardrail_span_fails_with_reason():
    state, _root = _start_trace()
    adapter.install()
    guardrail = _input_guardrail(triggered=True, reasoning="contains an SSN")

    asyncio.run(
        guardrail.run(
            SimpleNamespace(name="triage"), "my SSN is 123-45-6789", SimpleNamespace()
        )
    )

    span = [s for s in state.completed_spans if s.type == SpanType.GUARDRAIL][0]
    assert span.metadata["tripwire_triggered"] is True
    assert span.status == "failed"
    assert "Relevance Guardrail" in span.error_message
    assert span.output_data == {"reasoning": "contains an SSN"}


def test_output_guardrail_span_records_kind_and_checked_output():
    from agents.guardrail import GuardrailFunctionOutput, OutputGuardrail

    state, _root = _start_trace()
    adapter.install()

    async def _check(context, agent, agent_output):
        return GuardrailFunctionOutput(output_info=None, tripwire_triggered=False)

    guardrail = OutputGuardrail(guardrail_function=_check, name="Jailbreak Guardrail")
    asyncio.run(
        guardrail.run(SimpleNamespace(), SimpleNamespace(name="triage"), "the answer")
    )

    span = [s for s in state.completed_spans if s.type == SpanType.GUARDRAIL][0]
    assert span.metadata["guardrail_kind"] == "output"
    assert span.input_data == {"text": "the answer"}


def test_runner_call_inside_a_guardrail_nests_under_the_guardrail_span():
    """The guardrail's own LLM work must hang off the guardrail span — that it
    hung off the run root as an anonymous `llm` span is MHQ-1024 itself."""
    state, root_span = _start_trace()
    guardrail_span = SpanData(
        trace_id=state.trace_id,
        type=SpanType.GUARDRAIL,
        name="guardrail.Relevance Guardrail",
    )
    token = adapter._guardrail_span.set(guardrail_span)
    try:
        _state, _holder, _cm, chosen_root, _hooks = adapter._prepare_call(
            SimpleNamespace(name="guardrail-agent"), "some input", {}
        )
    finally:
        adapter._guardrail_span.reset(token)

    assert chosen_root is guardrail_span
    # ...and outside a guardrail the run root is still the top-of-stack span.
    _s, _h, _c, outside_root, _hk = adapter._prepare_call(
        SimpleNamespace(name="triage"), "some input", {}
    )
    assert outside_root is root_span


def test_root_span_input_is_the_run_input_not_the_builtin():
    """`_prepare_call` once read `input` from the enclosing scope, where it
    resolved to the builtin — every root span carried it as input_data.
    Fixed by MHQ-1022; this pins it, since a guardrail's nested run reuses
    the same path to record the input that tripped it."""
    _state, root_span = _start_trace()
    adapter._prepare_call(SimpleNamespace(name="triage"), "book me a flight", {})
    assert root_span.input_data == {"text": "book me a flight"}


def test_guardrail_span_not_emitted_without_an_active_trace():
    _active_trace.set(None)
    adapter.install()
    guardrail = _input_guardrail(triggered=True)
    result = asyncio.run(
        guardrail.run(SimpleNamespace(name="triage"), "hello", SimpleNamespace())
    )
    # The guardrail still ran and returned its verdict — telemetry is optional.
    assert result.output.tripwire_triggered is True


def test_uninstall_restores_guardrail_run():
    from agents.guardrail import InputGuardrail, OutputGuardrail

    original_input = InputGuardrail.run
    original_output = OutputGuardrail.run

    adapter.install()
    assert InputGuardrail.run is not original_input

    adapter.uninstall()
    assert InputGuardrail.run is original_input
    assert OutputGuardrail.run is original_output
