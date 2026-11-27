"""MHQ-1880: model-client calls carry a W3C ``traceparent`` naming the call's
``llm`` span, and the OTLP export reuses that span id, so a gateway that
forwards the header produces a span the server can parent exactly."""

# ruff: noqa: SLF001
from __future__ import annotations

import asyncio
import uuid
from types import SimpleNamespace
from typing import Any

import pytest

from amund import tracing
from amund._internal.redaction import RedactionConfig
from amund._internal.span_ids import otel_span_id_for
from amund.adapters import anthropic as av_anthropic
from amund.adapters import openai as av_openai
from amund.adapters.langchain import MorseCallbackHandler
from amund.otel_provider import get_engine

pytestmark = pytest.mark.req("MHQ-1880")


@pytest.fixture
def engine():
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
    try:
        yield eng
    finally:
        eng.reset_for_testing()


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv("MORSE_PROPAGATE_TRACE_CONTEXT", raising=False)
    tracing._active_trace.set(None)
    yield
    tracing._active_trace.set(None)


def _parse(header: str) -> tuple[str, str, str]:
    version, trace_id, span_id, _flags = header.split("-")
    assert version == "00"
    return trace_id, span_id, _flags


def _exported_llm_spans(engine, trace_id: str) -> list[Any]:
    engine.flush()
    return [
        s
        for s in engine.recorder.spans
        if s.context.trace_id == int(trace_id, 16)
        and s.attributes.get("amund.span_type") == "llm"
    ]


def _header_of(call: dict[str, Any]) -> str:
    return call["extra_headers"]["traceparent"]


def _anthropic_response():
    return SimpleNamespace(
        id="msg_1",
        model="claude-haiku-4-5",
        usage=SimpleNamespace(
            input_tokens=10,
            output_tokens=5,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        ),
        content=[SimpleNamespace(type="text", text="hi")],
    )


class _Messages:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if kwargs.get("stream"):
            return _FakeStream()
        return _anthropic_response()


class _AMessages(_Messages):
    async def create(self, **kwargs: Any) -> Any:  # type: ignore[override]
        return super().create(**kwargs)


class _FakeStream:
    def __iter__(self):
        yield SimpleNamespace(
            type="message_start",
            message=SimpleNamespace(
                id="msg_s",
                model="claude-haiku-4-5",
                usage=SimpleNamespace(
                    input_tokens=7,
                    output_tokens=0,
                    cache_creation_input_tokens=0,
                    cache_read_input_tokens=0,
                ),
            ),
        )
        yield SimpleNamespace(
            type="message_delta",
            delta=SimpleNamespace(stop_reason="end_turn"),
            usage=SimpleNamespace(
                input_tokens=None,
                output_tokens=3,
                cache_creation_input_tokens=None,
                cache_read_input_tokens=None,
            ),
        )

    def close(self) -> None:
        pass


class _Client:
    def __init__(self) -> None:
        self.messages = _Messages()


def _assert_header_matches_export(engine, trace_id: str, header: str) -> None:
    h_trace, h_span, _ = _parse(header)
    assert h_trace == trace_id
    llm = _exported_llm_spans(engine, trace_id)
    assert len(llm) == 1
    assert format(llm[0].context.span_id, "016x") == h_span
    assert format(llm[0].context.trace_id, "032x") == h_trace


def test_otel_span_id_is_nonzero_16_hex_and_stable():
    assert otel_span_id_for("ab" * 16) == "ab" * 8
    assert otel_span_id_for("0" * 32) != "0" * 16
    weird = otel_span_id_for("not-hex")
    assert len(weird) == 16 and int(weird, 16) != 0
    assert otel_span_id_for("not-hex") == weird


def test_anthropic_wrap_direct_create_header_equals_exported_span_id(engine):
    client = av_anthropic.wrap(_Client())
    with tracing.run("agent") as _:
        trace_id = tracing.get_active_trace().trace_id
        client.messages.create(model="m", max_tokens=5, messages=[])
    call = client.messages._av_inner.calls[0]
    _assert_header_matches_export(engine, trace_id, _header_of(call))


def test_anthropic_async_wrap_direct_create(engine):
    class AsyncAnthropic:
        def __init__(self) -> None:
            self.messages = _AMessages()

    async_client = AsyncAnthropic()
    wrapped = av_anthropic.wrap(async_client)

    async def go() -> str:
        with tracing.run("agent"):
            tid = tracing.get_active_trace().trace_id
            await wrapped.messages.create(model="m", max_tokens=5, messages=[])
        return tid

    trace_id = asyncio.run(go())
    _assert_header_matches_export(
        engine, trace_id, _header_of(async_client.messages.calls[0])
    )


def test_anthropic_wrap_stream_create_header_equals_exported_span_id(engine):
    client = av_anthropic.wrap(_Client())
    with tracing.run("agent"):
        trace_id = tracing.get_active_trace().trace_id
        for _ in client.messages.create(model="m", messages=[], stream=True):
            pass
    _assert_header_matches_export(
        engine, trace_id, _header_of(client.messages._av_inner.calls[0])
    )


def test_anthropic_install_path_header_equals_exported_span_id(engine):
    inner = _Messages()
    patched = av_anthropic._wrap_create(lambda self, **kw: self.create(**kw))
    with tracing.run("agent"):
        trace_id = tracing.get_active_trace().trace_id
        patched(inner, model="m", max_tokens=5, messages=[])
    _assert_header_matches_export(engine, trace_id, _header_of(inner.calls[0]))


def test_install_path_stream_header_equals_exported_span_id(engine):
    inner = _Messages()
    patched = av_anthropic._wrap_create(lambda self, **kw: self.create(**kw))
    with tracing.run("agent"):
        trace_id = tracing.get_active_trace().trace_id
        for _ in patched(inner, model="m", messages=[], stream=True):
            pass
    _assert_header_matches_export(engine, trace_id, _header_of(inner.calls[0]))


def test_beta_messages_create_is_patched_like_messages(engine):
    """install() routes ``beta.messages.create`` through the same wrapper."""
    beta_inner = _Messages()
    patched = av_anthropic._wrap_create(lambda self, **kw: self.create(**kw))
    with tracing.run("agent"):
        trace_id = tracing.get_active_trace().trace_id
        patched(beta_inner, model="m", betas=["x"], messages=[])
    _assert_header_matches_export(engine, trace_id, _header_of(beta_inner.calls[0]))


def test_install_patches_beta_messages(monkeypatch):
    import sys
    import types

    def _mod(name: str) -> types.ModuleType:
        m = types.ModuleType(name)
        monkeypatch.setitem(sys.modules, name, m)
        return m

    class Messages:
        def create(self, **kw):  # noqa: ANN001, ANN201
            return None

        def stream(self, **kw):  # noqa: ANN001, ANN201
            return None

    class AsyncMessages:
        async def create(self, **kw):  # noqa: ANN001, ANN201
            return None

        def stream(self, **kw):  # noqa: ANN001, ANN201
            return None

    class BetaMessages(Messages):
        def create(self, **kw):  # noqa: ANN001, ANN201
            return None

    class BetaAsyncMessages(AsyncMessages):
        async def create(self, **kw):  # noqa: ANN001, ANN201
            return None

    root = _mod("anthropic")
    res = _mod("anthropic.resources")
    msgs = _mod("anthropic.resources.messages")
    beta = _mod("anthropic.resources.beta")
    beta_msgs = _mod("anthropic.resources.beta.messages")
    root.resources = res
    res.messages = msgs
    res.beta = beta
    beta.messages = beta_msgs
    msgs.Messages, msgs.AsyncMessages = Messages, AsyncMessages
    beta_msgs.Messages, beta_msgs.AsyncMessages = BetaMessages, BetaAsyncMessages
    orig_beta = BetaMessages.create
    try:
        assert av_anthropic.install() is True
        assert BetaMessages.create is not orig_beta
        assert BetaMessages.create.__wrapped__ is orig_beta
    finally:
        av_anthropic.uninstall()
    assert BetaMessages.create is orig_beta


def test_langchain_handler_llm_span_is_the_header_target_and_only_llm_span(engine):
    """ChatAnthropic -> ``with_raw_response.create`` hits the patched class
    method while the handler's llm span is open: the header names that span and
    no second llm span is recorded."""
    inner = _Messages()
    patched = av_anthropic._wrap_create(lambda self, **kw: self.create(**kw))
    handler = MorseCallbackHandler("lc-agent")
    run_id = uuid.uuid4()
    with tracing.run("lc-agent"):
        trace_id = tracing.get_active_trace().trace_id
        handler.on_llm_start(
            {"id": ["langchain", "chat_models", "ChatAnthropic"]},
            ["hi"],
            run_id=run_id,
            invocation_params={"model": "claude-haiku-4-5"},
        )
        handler_span_id = tracing.get_active_trace().span_stack[-1].span_id
        patched(inner, model="m", max_tokens=5, messages=[])
        handler.on_llm_end(
            SimpleNamespace(generations=[[SimpleNamespace(text="x")]], llm_output={}),
            run_id=run_id,
        )
    header = _header_of(inner.calls[0])
    assert _parse(header)[1] == otel_span_id_for(handler_span_id)
    _assert_header_matches_export(engine, trace_id, header)


def test_openai_chat_create_header_equals_exported_span_id(engine):
    calls: list[dict[str, Any]] = []

    class Completions:
        def create(self, **kwargs: Any) -> Any:
            calls.append(kwargs)
            return SimpleNamespace(
                id="chatcmpl-1",
                model="gpt-4o-mini",
                usage=SimpleNamespace(
                    prompt_tokens=3,
                    completion_tokens=2,
                    total_tokens=5,
                    prompt_tokens_details=SimpleNamespace(cached_tokens=0),
                    completion_tokens_details=SimpleNamespace(reasoning_tokens=0),
                ),
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="hi", tool_calls=None)
                    )
                ],
            )

    fake = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    client = av_openai.wrap(fake)
    with tracing.run("agent"):
        trace_id = tracing.get_active_trace().trace_id
        client.chat.completions.create(model="gpt-4o-mini", messages=[])
    _assert_header_matches_export(engine, trace_id, _header_of(calls[0]))


def test_caller_traceparent_preserved_and_headers_not_mutated(engine):
    client = av_anthropic.wrap(_Client())
    mine = {"TraceParent": "00-" + "1" * 32 + "-" + "2" * 16 + "-01", "x-a": "b"}
    snapshot = dict(mine)
    with tracing.run("agent"):
        client.messages.create(model="m", messages=[], extra_headers=mine)
    sent = client.messages._av_inner.calls[0]["extra_headers"]
    assert sent == snapshot
    assert mine == snapshot


def test_other_caller_headers_kept_and_original_dict_untouched(engine):
    client = av_anthropic.wrap(_Client())
    mine = {"x-a": "b"}
    with tracing.run("agent"):
        client.messages.create(model="m", messages=[], extra_headers=mine)
    sent = client.messages._av_inner.calls[0]["extra_headers"]
    assert sent["x-a"] == "b" and "traceparent" in sent
    assert mine == {"x-a": "b"}


@pytest.mark.parametrize("value", ["false", "0", "FALSE"])
def test_env_opt_out_sends_no_header(engine, monkeypatch, value):
    monkeypatch.setenv("MORSE_PROPAGATE_TRACE_CONTEXT", value)
    client = av_anthropic.wrap(_Client())
    with tracing.run("agent"):
        client.messages.create(model="m", messages=[])
    assert "extra_headers" not in client.messages._av_inner.calls[0]


def test_wrap_kwarg_opt_out_sends_no_header(engine):
    client = av_anthropic.wrap(_Client(), propagate_trace_context=False)
    with tracing.run("agent"):
        client.messages.create(model="m", messages=[])
    assert "extra_headers" not in client.messages._av_inner.calls[0]


def test_init_opt_out_sends_no_header(engine):
    import amund

    amund.init("mhq_test", propagate_trace_context=False)  # pragma: allowlist secret
    try:
        # init() may install the global patch, and then wrap() returns the
        # client untouched, so assert on the fake itself.
        fake = _Client()
        client = av_anthropic.wrap(fake)
        with tracing.run("agent"):
            client.messages.create(model="m", messages=[])
        assert "extra_headers" not in fake.messages.calls[0]
    finally:
        amund._reset_for_testing()


def test_no_active_trace_on_stream_sends_no_header():
    """A stream opened outside any trace has nowhere to point a header."""
    client = av_anthropic.wrap(_Client())
    client.messages.create(model="m", messages=[], stream=True)
    assert "extra_headers" not in client.messages._av_inner.calls[0]


def test_prepare_llm_call_no_trace_returns_inputs_unchanged():
    from amund.propagation import prepare_llm_call

    kwargs = {"model": "m"}
    out, span_id = prepare_llm_call(kwargs, own_span=True)
    assert out is kwargs and span_id is None
