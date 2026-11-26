"""MHQ-1922 and MHQ-1921 SDK follow-ups from tester round 2 (MHQ-1919).

MHQ-1922:
- ``init()`` without a key (or with an empty one) never raises: one warning,
  the SDK stays a no-op.
- A 401/403 on export says the key was rejected; a network failure keeps the
  "could not reach" wording.

MHQ-1921:
- Failed ``llm`` spans carry the provider's HTTP status and error text
  (OpenAI, Anthropic, Google GenAI).
- Inferred tool spans end before the next model call's ``llm`` span starts.
- The root span's error message starts with the exception type.
- A plain-Python tool wrapped in ``span(name, "tool")`` that raises leaves a
  failed ``tool`` span.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

import pytest

import amund
from amund import otel_provider, tracing
from amund._internal import diagnostics

httpx = pytest.importorskip("httpx")


@pytest.fixture(autouse=True)
def _reset():
    amund._reset_for_testing()
    diagnostics._reset_for_testing()
    yield
    amund._reset_for_testing()
    diagnostics._reset_for_testing()


def _warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]


# ---------------------------------------------------------------------------
# MHQ-1922.1 — init() never raises without a key
# ---------------------------------------------------------------------------


@pytest.mark.req("MHQ-1922")
def test_init_without_any_key_warns_once_and_stays_a_noop(monkeypatch, caplog):
    monkeypatch.delenv("AMUND_API_KEY", raising=False)
    monkeypatch.delenv("MORSE_API_KEY", raising=False)
    with caplog.at_level(logging.WARNING, logger="amund"):
        amund.init()  # must not raise
        amund.init()
        amund.track(agent_id="a", run_data={})
    assert amund._client is None
    warnings = _warnings(caplog)
    assert warnings == ["amund: no API key (set AMUND_API_KEY); tracing is off"]


@pytest.mark.req("MHQ-1922")
@pytest.mark.parametrize("bad_key", ["", "   ", 123])
def test_init_with_empty_or_invalid_explicit_key_warns(monkeypatch, caplog, bad_key):
    monkeypatch.delenv("AMUND_API_KEY", raising=False)
    monkeypatch.delenv("MORSE_API_KEY", raising=False)
    with caplog.at_level(logging.WARNING, logger="amund"):
        amund.init(api_key=bad_key)  # type: ignore[arg-type]
    assert amund._client is None
    (warning,) = _warnings(caplog)
    assert "AMUND_API_KEY" in warning
    assert "tracing is off" in warning


@pytest.mark.req("MHQ-1922")
def test_tracing_without_a_key_runs_user_code_normally(monkeypatch):
    monkeypatch.delenv("AMUND_API_KEY", raising=False)
    monkeypatch.delenv("MORSE_API_KEY", raising=False)
    amund.init()

    @tracing.trace_agent("agent")
    def agent() -> str:
        return "ok"

    assert agent() == "ok"


# ---------------------------------------------------------------------------
# MHQ-1922.2 — rejected key vs unreachable API
# ---------------------------------------------------------------------------


@pytest.mark.req("MHQ-1922")
def test_auth_drop_message_matches_typescript_wording(caplog):
    with caplog.at_level(logging.WARNING, logger="amund"):
        diagnostics.record_drop(diagnostics.REASON_AUTH_REJECTED, "span", 3, status=401)
    assert _warnings(caplog) == [
        "amund: Amund rejected the API key (401): 3 span(s) dropped. Check "
        "AMUND_API_KEY (or MORSE_API_KEY) or get a new key in Settings → API keys."
    ]


@pytest.mark.req("MHQ-1922")
@pytest.mark.parametrize(
    "reason", [diagnostics.REASON_RETRY_EXHAUSTED, diagnostics.REASON_SEND_FAILED]
)
def test_network_failure_keeps_unreachable_wording(caplog, reason):
    with caplog.at_level(logging.WARNING, logger="amund"):
        diagnostics.record_drop(reason, "span", 2)
    (warning,) = _warnings(caplog)
    assert "could not reach the Amund API" in warning
    assert "rejected the API key" not in warning


def _otlp_exporter(status: int) -> Any:
    exporter = otel_provider._DropReportingOTLPSpanExporter(
        endpoint="http://localhost:1/api/v1/otlp/v1/traces",
        headers={"Authorization": "Bearer test"},
        timeout=0.001,
    )

    def _fake_post(**_kwargs: Any) -> Any:
        return SimpleNamespace(
            status_code=status, ok=False, reason="stub", text="", headers={}
        )

    exporter._session = SimpleNamespace(post=_fake_post, close=lambda: None)
    return exporter


def _readable_span() -> Any:
    from opentelemetry.sdk.trace import ReadableSpan
    from opentelemetry.trace import SpanContext, TraceFlags

    return ReadableSpan(
        name="s",
        context=SpanContext(
            trace_id=0x1234567890ABCDEF1234567890ABCDEF,
            span_id=0x1234567890ABCDEF,
            is_remote=False,
            trace_flags=TraceFlags(TraceFlags.SAMPLED),
        ),
        start_time=1,
        end_time=2,
    )


@pytest.mark.req("MHQ-1922")
@pytest.mark.parametrize("status", [401, 403])
def test_otlp_export_rejected_key_says_rejected(caplog, status):
    exporter = _otlp_exporter(status)
    with caplog.at_level(logging.WARNING, logger="amund"):
        exporter.export([_readable_span(), _readable_span()])
    text = " ".join(_warnings(caplog))
    assert f"amund: Amund rejected the API key ({status}): 2 span(s) dropped." in text
    assert "could not reach" not in text


@pytest.mark.req("MHQ-1922")
def test_otlp_export_connection_refused_says_unreachable(caplog, monkeypatch):
    import requests

    exporter = _otlp_exporter(200)

    def _refuse(**_kwargs: Any) -> Any:
        raise requests.exceptions.ConnectionError("refused")

    exporter._session = SimpleNamespace(post=_refuse, close=lambda: None)
    with caplog.at_level(logging.WARNING, logger="amund"):
        exporter.export([_readable_span()])
    text = " ".join(_warnings(caplog))
    assert "could not reach the Amund API" in text
    assert "rejected the API key" not in text


# ---------------------------------------------------------------------------
# MHQ-1921 — root span error message, tool span on a raising plain tool
# ---------------------------------------------------------------------------


class NotFoundError(Exception):
    pass


@pytest.mark.req("MHQ-1921")
def test_root_error_message_starts_with_exception_type_decorator():
    @tracing.trace_agent("agent")
    def agent() -> None:
        raise NotFoundError("The model `gpt-9` does not exist")

    with pytest.raises(NotFoundError):
        agent()
    trace = tracing._get_last_trace()
    assert trace["trace"]["error_message"] == (
        "NotFoundError: The model `gpt-9` does not exist"
    )


@pytest.mark.req("MHQ-1921")
def test_root_error_message_starts_with_exception_type_context_manager():
    with pytest.raises(NotFoundError), tracing.run("agent"):
        raise NotFoundError("The model `gpt-9` does not exist")
    trace = tracing._get_last_trace()
    assert trace["trace"]["error_message"] == (
        "NotFoundError: The model `gpt-9` does not exist"
    )


@pytest.mark.req("MHQ-1921")
def test_root_error_message_does_not_double_the_type():
    assert tracing.root_error_message(ValueError("ValueError: x")) == "ValueError: x"
    assert tracing.root_error_message(KeyError()) == "KeyError"


@pytest.mark.req("MHQ-1921")
def test_raising_plain_python_tool_in_span_leaves_failed_tool_span():
    def lookup_order(order_id: str) -> str:
        with tracing.span("lookup_order", "tool", input_data={"order_id": order_id}):
            raise LookupError(f"order {order_id} not found")

    with pytest.raises(LookupError), tracing.run("agent"):
        lookup_order("A-7")
    trace = tracing._get_last_trace()
    (tool,) = [s for s in trace["spans"] if s["type"] == "tool"]
    assert tool["name"] == "lookup_order"
    assert tool["status"] == "failed"
    assert tool["error_message"] == "order A-7 not found"
    assert tool["exception_type"] == "LookupError"


# ---------------------------------------------------------------------------
# MHQ-1921 — provider status + error text on failed llm spans
# ---------------------------------------------------------------------------


def _llm_spans() -> list[dict[str, Any]]:
    trace = tracing._get_last_trace()
    assert trace is not None
    return [s for s in trace["spans"] if s["type"] == "llm"]


_NOT_FOUND_TEXT = "The model `gpt-9` does not exist or you do not have access to it."


@pytest.mark.req("MHQ-1921")
def test_openai_404_records_status_and_provider_error():
    openai = pytest.importorskip("openai")
    from amund.adapters import openai as av_openai

    token = tracing._active_trace.set(None)
    av_openai.uninstall()
    av_openai.install()
    try:
        client = openai.OpenAI(
            api_key="sk-test-not-a-real-key",  # pragma: allowlist secret
            base_url="https://openai.test/v1",
            max_retries=0,
            http_client=httpx.Client(
                transport=httpx.MockTransport(
                    lambda r: httpx.Response(
                        404,
                        json={
                            "error": {
                                "message": _NOT_FOUND_TEXT,
                                "type": "invalid_request_error",
                                "code": "model_not_found",
                            }
                        },
                    )
                )
            ),
        )
        with pytest.raises(openai.NotFoundError), tracing.run("agent"):
            client.chat.completions.create(
                model="gpt-9", messages=[{"role": "user", "content": "hi"}]
            )
    finally:
        av_openai.uninstall()
        tracing._active_trace.reset(token)
    (llm,) = _llm_spans()
    assert llm["status"] == "failed"
    assert llm["metadata"]["http.status_code"] == 404
    assert llm["metadata"]["provider_error"] == _NOT_FOUND_TEXT
    trace = tracing._get_last_trace()
    assert trace["trace"]["error_message"].startswith("NotFoundError: ")


@pytest.mark.req("MHQ-1921")
def test_anthropic_404_records_status_and_provider_error():
    anthropic = pytest.importorskip("anthropic")
    from amund.adapters import anthropic as av_anthropic

    token = tracing._active_trace.set(None)
    av_anthropic.uninstall()
    av_anthropic.install()
    try:
        client = anthropic.Anthropic(
            api_key="sk-ant-test-not-a-real-key",  # pragma: allowlist secret
            base_url="https://anthropic.test",
            max_retries=0,
            http_client=httpx.Client(
                transport=httpx.MockTransport(
                    lambda r: httpx.Response(
                        404,
                        json={
                            "type": "error",
                            "error": {
                                "type": "not_found_error",
                                "message": "model: claude-9",
                            },
                        },
                    )
                )
            ),
        )
        with pytest.raises(anthropic.NotFoundError), tracing.run("agent"):
            client.messages.create(
                model="claude-9",
                max_tokens=10,
                messages=[{"role": "user", "content": "hi"}],
            )
    finally:
        av_anthropic.uninstall()
        tracing._active_trace.reset(token)
    (llm,) = _llm_spans()
    assert llm["status"] == "failed"
    assert llm["metadata"]["http.status_code"] == 404
    assert "model: claude-9" in llm["metadata"]["provider_error"]


@pytest.mark.req("MHQ-1921")
def test_google_genai_404_records_status_and_provider_error():
    genai = pytest.importorskip("google.genai")
    from google.genai import types

    from amund.adapters import google_genai as av_genai

    token = tracing._active_trace.set(None)
    av_genai.uninstall()
    assert av_genai.install() is True
    try:
        client = genai.Client(
            api_key="test-not-a-real-key",  # pragma: allowlist secret
            http_options=types.HttpOptions(
                base_url="https://genai.test",
                httpx_client=httpx.Client(
                    transport=httpx.MockTransport(
                        lambda r: httpx.Response(
                            404,
                            json={
                                "error": {
                                    "code": 404,
                                    "message": "models/gemini-9 is not found",
                                    "status": "NOT_FOUND",
                                }
                            },
                        )
                    )
                ),
            ),
        )
        with pytest.raises(genai.errors.ClientError), tracing.run("agent"):
            client.models.generate_content(model="gemini-9", contents="hi")
    finally:
        av_genai.uninstall()
        tracing._active_trace.reset(token)
    (llm,) = _llm_spans()
    assert llm["status"] == "failed"
    assert llm["metadata"]["http.status_code"] == 404
    assert llm["metadata"]["provider_error"] == "models/gemini-9 is not found"
    assert llm["metadata"]["provider_error_status"] == "NOT_FOUND"


def test_provider_error_metadata_ignores_plain_exceptions():
    assert tracing.provider_error_metadata(RuntimeError("boom")) == {}


# ---------------------------------------------------------------------------
# MHQ-1921 — inferred tool spans end before the next model call starts
# ---------------------------------------------------------------------------


def _chat_body(content: str | None, tool_calls: list[dict[str, Any]] | None = None):
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1,
        "model": "gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": "tool_calls" if tool_calls else "stop",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    }


@pytest.mark.req("MHQ-1921")
def test_openai_inferred_tool_span_ends_before_next_llm_span_starts():
    openai = pytest.importorskip("openai")
    from amund.adapters import openai as av_openai

    tool_calls = [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "get_weather", "arguments": "{}"},
        }
    ]
    bodies = iter([_chat_body(None, tool_calls), _chat_body("Sunny")])
    token = tracing._active_trace.set(None)
    av_openai.uninstall()
    av_openai.install()
    try:
        client = openai.OpenAI(
            api_key="sk-test-not-a-real-key",  # pragma: allowlist secret
            base_url="https://openai.test/v1",
            max_retries=0,
            http_client=httpx.Client(
                transport=httpx.MockTransport(
                    lambda r: httpx.Response(200, json=next(bodies))
                )
            ),
        )
        messages: list[dict[str, Any]] = [{"role": "user", "content": "weather?"}]
        with tracing.run("agent"):
            first = client.chat.completions.create(
                model="gpt-4o-mini", messages=messages
            )
            messages.append(first.choices[0].message.model_dump(exclude_none=True))
            messages.append(
                {"role": "tool", "tool_call_id": "call_1", "content": "Sun"}
            )
            client.chat.completions.create(model="gpt-4o-mini", messages=messages)
    finally:
        av_openai.uninstall()
        tracing._active_trace.reset(token)
    trace = tracing._get_last_trace()
    (tool,) = [s for s in trace["spans"] if s["type"] == "tool"]
    llms = sorted(_llm_spans(), key=lambda s: s["started_at"])
    assert len(llms) == 2
    assert tool["ended_at"] <= llms[1]["started_at"]
