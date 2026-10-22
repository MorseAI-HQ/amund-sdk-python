"""MHQ-1415: Anthropic streaming must produce one ``llm`` span per call.

``messages.stream()`` posts its own request and never calls ``create()``, and
``create(stream=True)`` returns an event iterator rather than a ``Message``.
Before the fix the first recorded nothing and the second recorded a span with
no model, tokens or cost.

The fake-based tests cover ``wrap()`` without the ``anthropic`` package. The
last section runs the real SDK (skipped when it isn't installed) against a
mocked SSE response, covering ``install()``, the zero-code path.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any

import pytest

from morse_ai.adapters import anthropic as av_anthropic
from morse_ai.spans import SpanType


@pytest.fixture
def captured_traces():
    """Capture finished trace entries (same shape as test_anthropic_wrap)."""
    from morse_ai import tracing as _tracing

    captured: list[dict[str, Any]] = []
    original = _tracing._finish_trace

    def _capture(entry: dict[str, Any]) -> None:
        _tracing._capture_last_trace(entry)
        captured.append(entry)

    _tracing._finish_trace = _capture
    try:
        yield captured
    finally:
        _tracing._finish_trace = original


def _llm_spans(captured: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        s
        for entry in captured
        for s in entry["spans"]
        if s["type"] == SpanType.LLM.value
    ]


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


def _final_message(*, stop_reason: str | None = "end_turn") -> SimpleNamespace:
    return SimpleNamespace(
        model="claude-sonnet-5",
        stop_reason=stop_reason,
        usage=SimpleNamespace(
            input_tokens=12,
            output_tokens=7,
            cache_creation_input_tokens=50,
            cache_read_input_tokens=100,
        ),
        content=[SimpleNamespace(type="text", text="Hello")],
    )


class _FakeStream:
    def __init__(self, snapshot: Any) -> None:
        self._snapshot = snapshot

    @property
    def current_message_snapshot(self) -> Any:
        assert self._snapshot is not None
        return self._snapshot


class _FakeManager:
    def __init__(self, snapshot: Any) -> None:
        self.stream = _FakeStream(snapshot)
        self.exited = False

    def __enter__(self) -> _FakeStream:
        return self.stream

    def __exit__(self, *exc: Any) -> None:
        self.exited = True


class _FakeAsyncManager(_FakeManager):
    async def __aenter__(self) -> _FakeStream:
        return self.stream

    async def __aexit__(self, *exc: Any) -> None:
        self.exited = True


def _events() -> list[SimpleNamespace]:
    usage_start = SimpleNamespace(
        input_tokens=12,
        output_tokens=1,
        cache_creation_input_tokens=50,
        cache_read_input_tokens=100,
    )
    return [
        SimpleNamespace(
            type="message_start",
            message=SimpleNamespace(model="claude-sonnet-5", usage=usage_start),
        ),
        SimpleNamespace(
            type="content_block_start",
            index=0,
            content_block=SimpleNamespace(type="text", name=None),
        ),
        SimpleNamespace(
            type="content_block_delta",
            index=0,
            delta=SimpleNamespace(type="text_delta", text="Hel"),
        ),
        SimpleNamespace(
            type="content_block_delta",
            index=0,
            delta=SimpleNamespace(type="text_delta", text="lo"),
        ),
        SimpleNamespace(
            type="message_delta",
            delta=SimpleNamespace(stop_reason="end_turn"),
            usage=SimpleNamespace(output_tokens=7),
        ),
        SimpleNamespace(type="message_stop"),
    ]


class _FakeRawStream:
    def __init__(self, events: list[Any]) -> None:
        self._events = events
        self.closed = False

    def __iter__(self):
        yield from self._events

    def close(self) -> None:
        self.closed = True


class _FakeAsyncRawStream(_FakeRawStream):
    async def __aiter__(self):
        for event in self._events:
            yield event

    async def close(self) -> None:
        self.closed = True


class _SyncMessages:
    def __init__(self, manager: Any = None, raw: Any = None) -> None:
        self._manager = manager
        self._raw = raw

    def stream(self, **kwargs: Any) -> Any:
        return self._manager

    def create(self, **kwargs: Any) -> Any:
        assert kwargs.get("stream") is True
        return self._raw


class _AsyncMessages(_SyncMessages):
    def stream(self, **kwargs: Any) -> Any:  # not a coroutine, like the SDK
        return self._manager

    async def create(self, **kwargs: Any) -> Any:
        assert kwargs.get("stream") is True
        return self._raw


class _SyncClient:
    def __init__(self, messages: Any) -> None:
        self.messages = messages


class AsyncAnthropic:  # name prefix drives wrap()'s async detection
    def __init__(self, messages: Any) -> None:
        self.messages = messages


_KW = {
    "model": "claude-sonnet-5",
    "max_tokens": 64,
    "messages": [{"role": "user", "content": "hi"}],
}


def _assert_full_span(span: dict[str, Any], name: str) -> None:
    assert span["name"] == name
    assert span["model"] == "claude-sonnet-5"
    assert span["output_tokens"] == 7
    assert span["cached_input_tokens"] == 100
    assert span["cache_write_tokens"] == 50
    assert span["input_tokens"]  # rebuilt, not the old NULL span
    assert span["status"] == "completed"
    assert span["metadata"]["streamed"] is True
    assert "stream_incomplete" not in span["metadata"]


# ---------------------------------------------------------------------------
# wrap(): messages.stream
# ---------------------------------------------------------------------------


@pytest.mark.req("MHQ-1415")
def test_wrap_sync_messages_stream_records_one_llm_span(captured_traces):
    manager = _FakeManager(_final_message())
    client = av_anthropic.wrap(_SyncClient(_SyncMessages(manager=manager)))

    with client.messages.stream(**_KW) as stream:
        assert stream is manager.stream  # the SDK's own stream object, untouched

    assert manager.exited
    spans = _llm_spans(captured_traces)
    assert len(spans) == 1
    _assert_full_span(spans[0], "anthropic.messages.stream")


@pytest.mark.req("MHQ-1415")
def test_wrap_async_messages_stream_records_one_llm_span(captured_traces):
    manager = _FakeAsyncManager(_final_message())
    client = av_anthropic.wrap(AsyncAnthropic(_AsyncMessages(manager=manager)))

    async def _go() -> None:
        async with client.messages.stream(**_KW) as stream:
            assert stream is manager.stream

    asyncio.run(_go())
    spans = _llm_spans(captured_traces)
    assert len(spans) == 1
    _assert_full_span(spans[0], "anthropic.messages.stream")


@pytest.mark.req("MHQ-1415")
def test_abandoned_stream_is_marked_incomplete(captured_traces):
    manager = _FakeManager(_final_message(stop_reason=None))
    client = av_anthropic.wrap(_SyncClient(_SyncMessages(manager=manager)))

    with client.messages.stream(**_KW):
        pass  # consumer left before the final event

    [span] = _llm_spans(captured_traces)
    assert span["status"] == "completed"
    assert span["metadata"]["stream_incomplete"] is True


@pytest.mark.req("MHQ-1415")
def test_error_inside_stream_records_failed_span_and_reraises(captured_traces):
    manager = _FakeManager(None)  # no event arrived before the failure
    client = av_anthropic.wrap(_SyncClient(_SyncMessages(manager=manager)))

    with pytest.raises(RuntimeError, match="boom"), client.messages.stream(**_KW):
        raise RuntimeError("boom")

    [span] = _llm_spans(captured_traces)
    assert span["status"] == "failed"
    assert "RuntimeError: boom" in span["error_message"]
    assert (
        span["model"] == "claude-sonnet-5"
    )  # requested model, though no event arrived


# ---------------------------------------------------------------------------
# wrap(): create(stream=True)
# ---------------------------------------------------------------------------


@pytest.mark.req("MHQ-1415")
def test_wrap_sync_create_stream_rebuilds_usage_from_events(captured_traces):
    client = av_anthropic.wrap(
        _SyncClient(_SyncMessages(raw=_FakeRawStream(_events())))
    )

    seen = [e.type for e in client.messages.create(stream=True, **_KW)]

    assert seen[0] == "message_start" and seen[-1] == "message_stop"
    [span] = _llm_spans(captured_traces)
    _assert_full_span(span, "anthropic.messages.create")
    assert span["output_data"] == {"content": "Hello"}


@pytest.mark.req("MHQ-1415")
def test_wrap_async_create_stream_rebuilds_usage_from_events(captured_traces):
    raw = _FakeAsyncRawStream(_events())
    client = av_anthropic.wrap(AsyncAnthropic(_AsyncMessages(raw=raw)))

    async def _go() -> int:
        stream = await client.messages.create(stream=True, **_KW)
        return len([e async for e in stream])

    assert asyncio.run(_go()) == len(_events())
    [span] = _llm_spans(captured_traces)
    _assert_full_span(span, "anthropic.messages.create")


@pytest.mark.req("MHQ-1415")
def test_create_stream_early_break_records_once(captured_traces):
    raw = _FakeRawStream(_events())
    client = av_anthropic.wrap(_SyncClient(_SyncMessages(raw=raw)))

    stream = client.messages.create(stream=True, **_KW)
    for event in stream:
        if event.type == "content_block_delta":
            break
    stream.close()

    assert raw.closed
    [span] = _llm_spans(captured_traces)
    assert span["model"] == "claude-sonnet-5"
    assert span["metadata"]["stream_incomplete"] is True


# ---------------------------------------------------------------------------
# install() against the real anthropic SDK (skipped when not installed)
# ---------------------------------------------------------------------------

_SSE_EVENTS = [
    {
        "type": "message_start",
        "message": {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-5",
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {
                "input_tokens": 12,
                "output_tokens": 1,
                "cache_creation_input_tokens": 50,
                "cache_read_input_tokens": 100,
            },
        },
    },
    {
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "text", "text": ""},
    },
    {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "text_delta", "text": "Hello"},
    },
    {"type": "content_block_stop", "index": 0},
    {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
        "usage": {"output_tokens": 7},
    },
    {"type": "message_stop"},
]
_SSE_BODY = "".join(
    f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in _SSE_EVENTS
).encode()


@pytest.fixture
def real_anthropic():
    anthropic = pytest.importorskip("anthropic")
    httpx = pytest.importorskip("httpx")

    def _handler(request: Any) -> Any:
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=_SSE_BODY
        )

    av_anthropic.install()
    try:
        yield anthropic, httpx, _handler
    finally:
        av_anthropic.uninstall()


@pytest.mark.req("MHQ-1415")
def test_install_traces_real_sync_stream_and_create_stream(
    real_anthropic, captured_traces
):
    anthropic, httpx, handler = real_anthropic
    client = anthropic.Anthropic(
        api_key="test",  # pragma: allowlist secret
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    with client.messages.stream(**_KW) as stream:
        assert stream.get_final_message().content[0].text == "Hello"
    events = list(client.messages.create(stream=True, **_KW))

    assert events[-1].type == "message_stop"
    spans = _llm_spans(captured_traces)
    assert [s["name"] for s in spans] == [
        "anthropic.messages.stream",
        "anthropic.messages.create",
    ]
    for span in spans:
        _assert_full_span(span, span["name"])


@pytest.mark.req("MHQ-1415")
def test_install_traces_real_async_stream_and_wrap_does_not_double(
    real_anthropic, captured_traces
):
    anthropic, httpx, handler = real_anthropic
    client = av_anthropic.wrap(
        anthropic.AsyncAnthropic(
            api_key="test",  # pragma: allowlist secret
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )
    )

    async def _go() -> None:
        async with client.messages.stream(**_KW) as stream:
            await stream.get_final_message()
        async for _ in await client.messages.create(stream=True, **_KW):
            pass

    asyncio.run(_go())
    spans = _llm_spans(captured_traces)
    assert [s["name"] for s in spans] == [
        "anthropic.messages.stream",
        "anthropic.messages.create",
    ]
