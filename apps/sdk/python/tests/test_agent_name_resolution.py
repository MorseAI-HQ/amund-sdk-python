"""MHQ-1100: per-trace agent-name resolution; no env service.name identity."""

from __future__ import annotations

import sys
import uuid

import pytest

import morse_ai
from morse_ai import tracing
from morse_ai._internal import agent_name as an
from morse_ai._internal.redaction import RedactionConfig
from morse_ai.otel_provider import get_engine

_ENV_KEYS = ("OTEL_SERVICE_NAME", "OTEL_RESOURCE_ATTRIBUTES")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in _ENV_KEYS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(sys, "argv", ["pytest"])
    monkeypatch.setattr(sys.modules["__main__"], "__spec__", None, raising=False)


@pytest.fixture
def engine():
    """Factory: env must be set BEFORE configure() builds the Resource."""
    engs = []

    def factory():
        eng = get_engine()
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
        engs.append(eng)
        return eng

    yield factory
    for eng in engs:
        eng.reset_for_testing()


def _root(eng, trace_id_hex):
    eng.provider.force_flush()
    tid = int(trace_id_hex, 16)
    spans = [s for s in eng.recorder.spans if s.context.trace_id == tid]
    assert spans
    return spans[0]


def test_run_agent_wins_over_process_service_name(monkeypatch, engine):
    monkeypatch.setenv("OTEL_SERVICE_NAME", "proc")
    eng = engine()
    with morse_ai.run("copilot") as r:
        trace_id = r.trace_id
    root = _root(eng, trace_id)
    assert root.attributes["morse.agent"] == "copilot"
    res = dict(root.resource.attributes)
    assert res.get("service.name") != "proc"
    assert res["morse.process.service_name"] == "proc"


def test_no_env_service_name_leaves_no_process_attr(engine):
    eng = engine()
    with morse_ai.run("copilot") as r:
        trace_id = r.trace_id
    root = _root(eng, trace_id)
    assert "morse.process.service_name" not in dict(root.resource.attributes)


def test_resource_attributes_service_name_is_demoted(monkeypatch, engine):
    monkeypatch.setenv("OTEL_RESOURCE_ATTRIBUTES", "service.name=viaattrs,x=y")
    eng = engine()
    with morse_ai.run("a") as r:
        trace_id = r.trace_id
    res = dict(_root(eng, trace_id).resource.attributes)
    assert res.get("service.name") != "viaattrs"
    assert res["morse.process.service_name"] == "viaattrs"


# --- resolver tiers -------------------------------------------------------


def test_tier_a_explicit_wins(monkeypatch):
    monkeypatch.setenv("OTEL_SERVICE_NAME", "proc")
    assert an.resolve_agent_name("x", framework="fw") == "x"


def test_tier_b_framework_beats_env(monkeypatch):
    monkeypatch.setenv("OTEL_SERVICE_NAME", "proc")
    assert an.resolve_agent_name(None, framework="fw") == "fw"
    assert an.resolve_agent_name("  ", framework="fw") == "fw"


def test_tier_c_env_service_name(monkeypatch):
    monkeypatch.setenv("OTEL_SERVICE_NAME", "proc")
    assert an.resolve_agent_name(None) == "proc"


def test_tier_c_resource_attributes(monkeypatch):
    monkeypatch.setenv("OTEL_RESOURCE_ATTRIBUTES", "a=b, service.name=svc")
    assert an.resolve_agent_name(None) == "svc"


def test_tier_d_script_stem(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["/srv/app/my_bot.py"])
    assert an.resolve_agent_name(None) == "my_bot"


@pytest.mark.parametrize(
    "argv0",
    ["python", "-c", "", "manage.py", "gunicorn", "uvicorn", "celery", "pytest"],
)
def test_tier_d_skips_generic(monkeypatch, argv0):
    monkeypatch.setattr(sys, "argv", [argv0])
    assert an.resolve_agent_name(None) == "default-agent"


def test_tier_e_default():
    assert an.resolve_agent_name(None) == "default-agent"


def test_run_none_resolves(monkeypatch):
    monkeypatch.setenv("OTEL_SERVICE_NAME", "proc")
    with morse_ai.run(None) as r:
        assert tracing.get_active_trace().agent_name == "proc"
        assert r.trace_id


# --- LangGraph --------------------------------------------------------------


def _lg_trace(graph_name, agent_name=None):
    from morse_ai.adapters.langgraph import LangGraphCallbackHandler

    handler = LangGraphCallbackHandler(agent_name)
    run_id = uuid.uuid4()
    handler.on_chain_start(
        {"id": ["langgraph", "graph", "CompiledStateGraph"], "name": graph_name},
        {"messages": []},
        run_id=run_id,
        name=graph_name,
    )
    name = tracing.get_active_trace().agent_name
    handler.on_chain_end({}, run_id=run_id)
    return name


def test_langgraph_uses_graph_name():
    assert _lg_trace("my_graph") == "my_graph"


def test_langgraph_generic_default_is_not_explicit(monkeypatch):
    monkeypatch.setenv("OTEL_SERVICE_NAME", "proc")
    assert _lg_trace("LangGraph") == "proc"


def test_langgraph_explicit_beats_graph_name():
    assert _lg_trace("my_graph", agent_name="explicit") == "explicit"


def test_git_revision_rides_the_otel_resource(monkeypatch, engine):
    for var in ("GITHUB_SHA", "CI_COMMIT_SHA"):  # outrank Railway in CI
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("RAILWAY_GIT_COMMIT_SHA", "abc123")
    eng = engine()
    with morse_ai.run("a") as r:
        trace_id = r.trace_id
    assert dict(_root(eng, trace_id).resource.attributes)["git.revision"] == "abc123"
