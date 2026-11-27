"""Every OTLP attribute the SDK emits is under ``amund.*`` (no ``morse.*``).

Founder decision 2026-11-27: the former ``morse.<suffix>`` span/resource keys
are emitted as ``amund.<suffix>`` ONLY (no double emission). The server reads
``amund.*`` first and falls back to ``morse.*`` for older SDKs.
"""

from __future__ import annotations

import uuid

import pytest

import amund
from amund._internal.redaction import RedactionConfig
from amund.otel_provider import emit_trace_entry, get_engine

# old key -> new key; also rendered as a table in the README / CHANGELOG.
KEY_MAP = {
    "morse.span_type": "amund.span_type",
    "morse.cost_usd": "amund.cost_usd",
    "morse.input_data": "amund.input_data",
    "morse.output_data": "amund.output_data",
    "morse.context_segments": "amund.context_segments",
    "morse.metadata.topic": "amund.metadata.topic",
    "morse.agent": "amund.agent",
    "morse.agent_version": "amund.agent_version",
    "morse.identity_source": "amund.identity_source",
    "morse.app": "amund.app",
    "morse.thread_id": "amund.thread_id",
    "morse.user_id": "amund.user_id",
    "morse.user_metadata": "amund.user_metadata",
    "morse.outcome.success": "amund.outcome.success",
    "morse.outcome.label": "amund.outcome.label",
    "morse.record_key.order_id": "amund.record_key.order_id",
    "morse.framework": "amund.framework",
    "morse.source": "amund.source",
    "morse.prompt.name": "amund.prompt.name",
    "morse.prompt.version": "amund.prompt.version",
    "morse.prompt.link_source": "amund.prompt.link_source",
    "morse.exception_type": "amund.exception_type",
    "morse.error_message": "amund.error_message",
}


@pytest.fixture
def engine():
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
    try:
        yield eng
    finally:
        eng.reset_for_testing()


def _entry(trace_id: str) -> dict:
    span = {
        "span_id": uuid.uuid4().hex,
        "trace_id": trace_id,
        "parent_span_id": None,
        "type": "llm",
        "name": "root",
        "status": "failed",
        "started_at": "2026-01-01T00:00:00+00:00",
        "ended_at": "2026-01-01T00:00:01+00:00",
        "duration_ms": 1000,
        "cost_usd": 0.5,
        "input_data": "in",
        "output_data": "out",
        "context_segments": [{"kind": "system", "text": "s"}],
        "metadata": {"topic": "t"},
        "exception_type": "ValueError",
        "error_message": "boom",
    }
    return {
        "trace": {
            "trace_id": trace_id,
            "agent": "a1",
            "agent_version": "1.2.3",
            "status": "failed",
            "started_at": "2026-01-01T00:00:00+00:00",
            "metadata": {"identity_source": "explicit", "morse_app": "app1"},
            "thread_id": "th",
            "user_id": "u1",
            "user_metadata": {"plan": "pro"},
            "success": True,
            "outcome": "qualified",
            "record_keys": {"order_id": "5"},
        },
        "spans": [span],
        "resource": {"morse_ai.framework": "fw", "morse_ai.source": "src"},
        "prompts": [],
    }


def _root_attrs(engine, trace_id: str) -> dict:
    engine.flush()
    tid = int(trace_id, 16)
    spans = [s for s in engine.recorder.spans if s.context.trace_id == tid]
    assert spans
    return dict(spans[0].attributes)


def test_exact_amund_keys_and_no_morse_keys(engine):
    trace_id = uuid.uuid4().hex
    entry = _entry(trace_id)
    emit_trace_entry(entry)
    attrs = _root_attrs(engine, trace_id)
    # prompt attrs ride a prompt ref; covered separately below.
    expected = {v for k, v in KEY_MAP.items() if not k.startswith("morse.prompt.")}
    got = {k for k in attrs if k.startswith("amund.") or k.startswith("morse.")}
    assert got == expected
    assert not [k for k in attrs if k.startswith("morse.")]
    assert attrs["amund.outcome.success"] is True
    assert attrs["amund.agent"] == "a1"
    assert attrs["amund.app"] == "app1"
    assert attrs["amund.record_key.order_id"] == "5"


def test_prompt_keys_are_amund(engine):
    with amund.run("agent") as r:
        amund.set_prompt("triage", version=3)
    attrs = _root_attrs(engine, r.trace_id)
    assert attrs["amund.prompt.name"] == "triage"
    assert attrs["amund.prompt.version"] == 3
    assert not [k for k in attrs if k.startswith("morse.")]


def test_resource_attrs_are_amund(monkeypatch, engine):
    monkeypatch.setenv("OTEL_SERVICE_NAME", "proc")
    from amund import __version__

    eng = get_engine()
    eng.reset_for_testing()
    eng.configure(
        endpoint="http://127.0.0.1:1/api/v1/otlp/v1/traces",
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
    with amund.run("agent") as r:
        pass
    eng.flush()
    tid = int(r.trace_id, 16)
    root = next(s for s in eng.recorder.spans if s.context.trace_id == tid)
    res = dict(root.resource.attributes)
    assert res["amund.sdk.version"] == __version__
    assert res["amund.process.service_name"] == "proc"
    assert not [k for k in res if k.startswith("morse.")]


def test_batched_export_keeps_namespace(engine):
    """Several traces through the batch processor all carry amund.* only."""
    ids = [uuid.uuid4().hex for _ in range(5)]
    for tid in ids:
        emit_trace_entry(_entry(tid))
    engine.flush()
    spans = [s for s in engine.recorder.spans if f"{s.context.trace_id:032x}" in ids]
    assert len(spans) == 5
    for s in spans:
        assert "amund.agent" in s.attributes
        assert not [k for k in s.attributes if k.startswith("morse.")]
