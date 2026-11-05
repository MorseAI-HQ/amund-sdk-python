"""Strands Agents adapter (MHQ-133): Strands' own OTel spans reach Morse.

Real ``strands-agents`` with a scripted in-process model (no Bedrock, no
keys). Spans are captured through the Morse OTel engine's in-memory recorder,
so nothing is dialed.
"""

from __future__ import annotations

from typing import Any

import pytest

strands = pytest.importorskip("strands")

from opentelemetry import trace as trace_api  # noqa: E402
from opentelemetry.sdk.trace import Event, ReadableSpan, TracerProvider  # noqa: E402
from opentelemetry.sdk.trace.export import (  # noqa: E402
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from strands import Agent, tool  # noqa: E402
from strands.models.model import Model  # noqa: E402
from strands.telemetry import tracer as st_tracer  # noqa: E402

from morse_ai._internal import auto_detect  # noqa: E402
from morse_ai._internal.active_framework import get_llm_owner  # noqa: E402
from morse_ai._internal.redaction import RedactionConfig  # noqa: E402
from morse_ai.adapters import strands as av_strands  # noqa: E402
from morse_ai.otel_provider import _RedactingSpanExporter, get_engine  # noqa: E402


class _ScriptedModel(Model):
    """Replays canned turns; turn 1 may request a tool."""

    def __init__(self, turns: list[list[dict[str, Any]]]) -> None:
        self._turns = turns
        self.config = {"model_id": "us.amazon.nova-micro-v1:0"}
        self.owner_seen: list[str | None] = []

    def update_config(self, **model_config: Any) -> None:
        return None

    def get_config(self) -> Any:
        return {"model_id": "us.amazon.nova-micro-v1:0"}

    async def structured_output(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ARG002
        raise NotImplementedError
        yield  # pragma: no cover

    async def stream(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ARG002
        self.owner_seen.append(get_llm_owner())
        for event in self._turns.pop(0):
            yield event


def _text_turn(text: str) -> list[dict[str, Any]]:
    return [
        {"messageStart": {"role": "assistant"}},
        {"contentBlockStart": {"start": {}}},
        {"contentBlockDelta": {"delta": {"text": text}}},
        {"contentBlockStop": {}},
        {"messageStop": {"stopReason": "end_turn"}},
        {
            "metadata": {
                "usage": {"inputTokens": 40, "outputTokens": 5, "totalTokens": 45},
                "metrics": {"latencyMs": 3},
            }
        },
    ]


def _tool_turn() -> list[dict[str, Any]]:
    return [
        {"messageStart": {"role": "assistant"}},
        {
            "contentBlockStart": {
                "start": {"toolUse": {"toolUseId": "t1", "name": "get_price"}}
            }
        },
        {"contentBlockDelta": {"delta": {"toolUse": {"input": '{"ticker": "AAPL"}'}}}},
        {"contentBlockStop": {}},
        {"messageStop": {"stopReason": "tool_use"}},
        {
            "metadata": {
                "usage": {"inputTokens": 30, "outputTokens": 7, "totalTokens": 37},
                "metrics": {"latencyMs": 2},
            }
        },
    ]


@tool
def get_price(ticker: str) -> str:
    """Return the price of a ticker."""
    return f"{ticker}: 42"


@pytest.fixture
def engine(monkeypatch):
    eng = get_engine()
    eng.configure(
        endpoint="http://127.0.0.1:1/api/v1/otlp/v1/traces",  # never dialed
        api_key="av_test",  # pragma: allowlist secret
        sample_rate=1.0,
        schedule_delay_millis=50,
        max_queue_size=4096,
        max_export_batch_size=512,
        max_field_bytes=65536,
        timeout=1.0,
        redaction_config=RedactionConfig.build(enabled=True),
        capture_for_tests=True,
    )
    # A fresh Strands tracer singleton per test, and no real exporter sends.
    monkeypatch.setattr(st_tracer, "_tracer_instance", None)
    monkeypatch.setattr(eng, "exporter", None)
    av_strands.uninstall()
    assert av_strands.install() is True
    try:
        yield eng
    finally:
        av_strands.uninstall()
        eng.reset_for_testing()


def _recorded(eng: Any) -> list[ReadableSpan]:
    eng.provider.force_flush()
    return [
        s
        for s in eng.recorder.spans
        if s.instrumentation_scope.name.startswith("strands")
    ]


def _by_op(spans: list[ReadableSpan]) -> dict[str, list[ReadableSpan]]:
    out: dict[str, list[ReadableSpan]] = {}
    for s in spans:
        out.setdefault(str(s.attributes.get("gen_ai.operation.name")), []).append(s)
    return out


@pytest.mark.req("MHQ-133")
def test_zero_code_agent_spans_reach_the_morse_provider(engine):
    model = _ScriptedModel([_tool_turn(), _text_turn("AAPL is 42")])
    agent = Agent(model=model, tools=[get_price], name="pricer", callback_handler=None)
    result = agent("price of AAPL?")
    assert "42" in str(result)

    spans = _recorded(engine)
    ops = _by_op(spans)
    assert len(ops["invoke_agent"]) == 1
    assert ops["invoke_agent"][0].attributes["gen_ai.agent.name"] == "pricer"
    assert len(ops["chat"]) == 2
    assert ops["chat"][0].attributes["gen_ai.request.model"] == (
        "us.amazon.nova-micro-v1:0"
    )
    assert {s.attributes["gen_ai.usage.input_tokens"] for s in ops["chat"]} == {30, 40}
    assert [s.attributes["gen_ai.tool.name"] for s in ops["execute_tool"]] == [
        "get_price"
    ]
    # One trace, parented under the agent span.
    assert len({s.context.trace_id for s in spans}) == 1
    scope = spans[0].instrumentation_scope
    assert scope.name == "strands.telemetry.tracer"
    assert scope.version and scope.version != "unknown"
    # Resource is the Morse engine's (service.name pinned, not Strands').
    assert spans[0].resource.attributes["service.name"] == "unknown_service"


@pytest.mark.req("MHQ-133")
def test_model_call_marks_strands_as_llm_owner(engine):
    model = _ScriptedModel([_text_turn("hi")])
    Agent(model=model, callback_handler=None)("hi")
    assert model.owner_seen == ["strands"]
    assert get_llm_owner() is None


@pytest.mark.req("MHQ-133")
def test_failed_model_call_releases_owner_and_fails_span(engine):
    class _Boom(_ScriptedModel):
        async def stream(self, *args: Any, **kwargs: Any) -> Any:
            self.owner_seen.append(get_llm_owner())
            raise RuntimeError("bedrock down")
            yield  # pragma: no cover

    model = _Boom([])
    with pytest.raises(RuntimeError):
        Agent(model=model, callback_handler=None)("hi")
    assert get_llm_owner() is None
    chat = _by_op(_recorded(engine))["chat"]
    assert chat[0].status.status_code == trace_api.StatusCode.ERROR


@pytest.mark.req("MHQ-133")
def test_multiagent_span_is_typed_agent(engine):
    tracer = st_tracer.get_tracer()
    span = tracer.start_multiagent_span("task", "swarm")
    tracer.end_swarm_span(span, result="done")
    span.end()  # Strands ends multi-agent spans itself, after end_swarm_span
    (swarm,) = [
        s
        for s in _recorded(engine)
        if s.attributes.get("gen_ai.operation.name") == "invoke_swarm"
    ]
    assert swarm.attributes["morse.span_type"] == "agent"


@pytest.mark.req("MHQ-133")
def test_without_init_strands_keeps_its_default(engine, monkeypatch):
    monkeypatch.setattr(engine, "provider", None)
    span = st_tracer.get_tracer().start_model_invoke_span(messages=[])
    assert not span.is_recording()
    st_tracer.get_tracer().end_span_with_error(span, "x")


class _ListExporter(SpanExporter):
    def __init__(self) -> None:
        self.spans: list[ReadableSpan] = []

    def export(self, spans: Any) -> Any:
        self.spans.extend(spans)
        return SpanExportResult.SUCCESS


@pytest.mark.req("MHQ-133")
def test_app_owned_provider_keeps_its_spans_and_morse_gets_a_copy(engine, monkeypatch):
    app_exporter = _ListExporter()
    app_provider = TracerProvider()
    app_provider.add_span_processor(SimpleSpanProcessor(app_exporter))
    monkeypatch.setattr(trace_api, "get_tracer_provider", lambda: app_provider)
    morse_exporter = _ListExporter()
    monkeypatch.setattr(engine, "exporter", morse_exporter)

    tracer = st_tracer.get_tracer()
    span = tracer.start_model_invoke_span(messages=[], model_id="m")
    tracer.end_model_invoke_span(
        span,
        {"role": "assistant", "content": [{"text": "hi"}]},
        {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2},
        {"latencyMs": 1},
        "end_turn",
    )
    # An unrelated instrumentation on the same provider is NOT forwarded.
    app_provider.get_tracer("someone.else").start_span("other").end()
    app_provider.force_flush()

    assert {s.name for s in app_exporter.spans} == {"chat", "other"}
    assert [s.name for s in morse_exporter.spans] == ["chat"]
    assert morse_exporter.spans[0] is not app_exporter.spans[0]
    # Not on the Morse provider: no double export.
    assert _recorded(engine) == []


@pytest.mark.req("MHQ-133")
def test_redacting_exporter_redacts_span_event_attributes():
    inner = _ListExporter()
    exporter = _RedactingSpanExporter(inner, RedactionConfig.build(enabled=True))
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    span = provider.get_tracer("strands.telemetry.tracer").start_span("chat")
    span.add_event("gen_ai.user.message", {"content": "mail me at a@example.com"})
    span.end()
    (out,) = inner.spans
    (event,) = out.events
    assert isinstance(event, Event)
    assert "a@example.com" not in event.attributes["content"]


@pytest.mark.req("MHQ-133")
def test_install_is_idempotent_and_uninstall_restores():
    original = st_tracer.Tracer.__dict__["start_model_invoke_span"]
    av_strands.uninstall()
    assert av_strands.install() is True
    assert av_strands.install() is True
    assert st_tracer.Tracer.__dict__["start_model_invoke_span"].__wrapped__ is original
    av_strands.uninstall()
    assert st_tracer.Tracer.__dict__["start_model_invoke_span"] is original
    assert av_strands.is_installed() is False


@pytest.mark.req("MHQ-133")
def test_auto_detect_targets_strands():
    assert auto_detect._adapter_targets()["strands"] == "morse_ai.adapters.strands"
