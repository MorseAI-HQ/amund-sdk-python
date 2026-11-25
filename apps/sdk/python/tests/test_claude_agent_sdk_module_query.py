"""MHQ-1053 — the module-level ``claude_agent_sdk.query()`` must emit telemetry.

Before this ticket the adapter wrapped module-level ``query()`` only to capture
the prompt: it re-yielded every message untouched and emitted nothing. Because
``install()`` still logged ``adapter installed``, a customer reaching for a
public, documented entry point got an encouraging log line and zero spans —
worse than an outright error, which at least does not signal success.

Every test here drives ``_wrap_module_query`` end to end over a fake upstream
generator. All of them fail on the pre-MHQ-1053 wrapper, which is the point:
a test that passes on the broken behaviour would pin nothing.
"""

from __future__ import annotations

import asyncio

from amund import tracing
from amund.adapters import claude_agent_sdk as adapter


class TextBlock:
    def __init__(self, text: str):
        self.text = text


class AssistantMessage:
    def __init__(self, content: list, usage: dict | None = None, model: str = ""):
        self.content = content
        self.usage = usage or {}
        self.model = model


class ResultMessage:
    def __init__(self, is_error: bool = False, total_cost_usd: float | None = None):
        self.is_error = is_error
        self.total_cost_usd = total_cost_usd


def _usage(input_tokens=10, output_tokens=5):
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    }


def _assistant(text="Paris.", model="claude-haiku-4-5-20251001"):
    return AssistantMessage([TextBlock(text)], _usage(), model)


def _fake_query(messages):
    """Stand-in for upstream ``claude_agent_sdk.query`` — an async generator."""

    async def original(*args, **kwargs):
        for msg in messages:
            yield msg

    return original


def _drain(traced, **kwargs):
    """Run the wrapped generator to exhaustion, returning what it yielded."""

    async def go():
        return [msg async for msg in traced(**kwargs)]

    return asyncio.run(go())


def _llm_spans(trace):
    return [s for s in trace["spans"] if s.get("type") == "llm"]


# ---------------------------------------------------------------------------
# The bug: no trace at all
# ---------------------------------------------------------------------------


def test_module_query_outside_a_run_emits_a_trace_with_an_llm_span():
    traced = adapter._wrap_module_query(_fake_query([_assistant()]))

    yielded = _drain(traced, prompt="What is the capital of France?")

    # The caller still sees every upstream message, untouched.
    assert len(yielded) == 1
    trace = tracing._get_last_trace()
    assert trace is not None, "query() emitted no trace — MHQ-1053 regression"
    spans = _llm_spans(trace)
    assert len(spans) == 1
    assert spans[0]["model"] == "claude-haiku-4-5-20251001"
    # Cost is what makes the span worth having; a $0 span reads as a free call.
    assert spans[0]["cost_usd"] > 0


def test_module_query_captures_the_prompt_as_span_input():
    traced = adapter._wrap_module_query(_fake_query([_assistant()]))

    _drain(traced, prompt="What is the capital of France?")

    span = _llm_spans(tracing._get_last_trace())[0]
    assert span["input_data"]["messages"][0]["content"] == (
        "What is the capital of France?"
    )
    assert "Paris" in span["output_data"]["content"]


def test_module_query_inside_a_run_joins_that_trace_rather_than_opening_one():
    traced = adapter._wrap_module_query(_fake_query([_assistant()]))

    async def go():
        with tracing.run("caller-agent"):
            async for _ in traced(prompt="hi"):
                pass

    asyncio.run(go())

    trace = tracing._get_last_trace()
    # The caller's agent name proves we piggybacked instead of opening our own.
    assert trace["trace"]["agent"] == "caller-agent"
    assert len(_llm_spans(trace)) == 1


def test_module_query_result_message_records_aggregate_cost_not_outcome():
    traced = adapter._wrap_module_query(
        _fake_query(
            [_assistant(), ResultMessage(is_error=False, total_cost_usd=0.1234)]
        )
    )

    _drain(traced, prompt="hi")

    trace = tracing._get_last_trace()
    # MHQ-1914: the CLI exit status is not an outcome.
    assert trace["trace"].get("success") is None
    root = next(s for s in trace["spans"] if s.get("type") == "agent")
    assert root["metadata"]["sdk_total_cost_usd"] == 0.1234


def test_module_query_result_message_error_does_not_stamp_an_outcome():
    traced = adapter._wrap_module_query(
        _fake_query([_assistant(), ResultMessage(is_error=True)])
    )

    _drain(traced, prompt="hi")

    assert tracing._get_last_trace()["trace"].get("success") is None


def test_module_query_does_not_claim_the_callers_outcome():
    """A ResultMessage says how the CLI turn went, not how the caller's run did.

    Stamping success onto a trace we did not open would let one instrumented
    call decide the verdict for a request that is still in progress.
    """
    traced = adapter._wrap_module_query(
        _fake_query([_assistant(), ResultMessage(is_error=True)])
    )

    async def go():
        with tracing.run("caller-agent") as r:
            r.set_outcome(success=True)
            async for _ in traced(prompt="hi"):
                pass

    asyncio.run(go())

    assert tracing._get_last_trace()["trace"]["success"] is True


def test_module_query_finalizes_the_trace_when_the_consumer_raises():
    """A caller that breaks out mid-stream must still get its spans.

    The generator is closed rather than exhausted, so finalization has to hang
    off the exception path — otherwise an early ``break`` silently loses the
    whole trace, which is the same class of bug as the one this ticket fixes.
    """
    traced = adapter._wrap_module_query(
        _fake_query([_assistant(), _assistant("second")])
    )

    async def go():
        gen = traced(prompt="hi")
        async for _ in gen:
            break
        await gen.aclose()

    asyncio.run(go())

    trace = tracing._get_last_trace()
    assert trace is not None
    assert len(_llm_spans(trace)) == 1
