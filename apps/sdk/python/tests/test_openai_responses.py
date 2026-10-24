"""MHQ-1198: the OpenAI adapter traces ``responses.create`` (Responses API),
not only ``chat.completions.create``. Fakes only — no real API calls.
"""

from __future__ import annotations

import asyncio
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

import morse_ai
from morse_ai import tracing
from morse_ai.adapters import openai as av_openai


def _responses_response(text: str = "hello there", function_calls: Any = ()) -> Any:
    output = [
        SimpleNamespace(type="function_call", name=n, arguments=a)
        for n, a in function_calls
    ]
    return SimpleNamespace(
        model="gpt-4o-mini",
        output_text=text,
        output=output,
        usage=SimpleNamespace(
            input_tokens=30,
            output_tokens=12,
            total_tokens=42,
            input_tokens_details=None,
            output_tokens_details=None,
        ),
    )


class FakeResponses:
    def __init__(self, response: Any) -> None:
        self._response = response

    def create(self, *args: Any, **kwargs: Any) -> Any:
        return self._response


class FakeAsyncResponses:
    def __init__(self, response: Any) -> None:
        self._response = response

    async def create(self, *args: Any, **kwargs: Any) -> Any:
        return self._response


class FakeClient:
    def __init__(self, response: Any | None = None) -> None:
        self.responses = FakeResponses(response or _responses_response())


def _llm_spans() -> list[dict[str, Any]]:
    trace = tracing._get_last_trace()
    assert trace is not None
    return [s for s in trace["spans"] if s.get("type") == "llm"]


@pytest.fixture(autouse=True)
def _reset():
    from morse_ai.tracing import _active_trace

    _active_trace.set(None)
    morse_ai._reset_for_testing()
    av_openai.uninstall()
    yield
    _active_trace.set(None)
    morse_ai._reset_for_testing()
    av_openai.uninstall()


@pytest.mark.req("MHQ-1198")
def test_wrap_traces_responses_create_with_tokens():
    client = av_openai.wrap(FakeClient())
    with tracing.run("agent"):
        client.responses.create(model="gpt-4o-mini", input="hi")

    spans = _llm_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span["name"] == "openai.responses.create"
    assert span["model"] == "gpt-4o-mini"
    assert span["input_tokens"] == 30
    assert span["output_tokens"] == 12


@pytest.mark.req("MHQ-1198")
def test_wrap_responses_captures_input_instructions_and_output():
    client = av_openai.wrap(FakeClient())
    with tracing.run("agent"):
        client.responses.create(
            model="gpt-4o-mini",
            instructions="You are terse.",
            input=[
                {"role": "user", "content": "first"},
                {"type": "function_call_output", "call_id": "c1", "output": "x"},
            ],
        )

    span = _llm_spans()[0]
    # MHQ-1410: tool-loop items are kept so the input panel shows the round trip.
    assert span["input_data"] == {
        "messages": [
            {"role": "system", "content": "You are terse."},
            {"role": "user", "content": "first"},
            {"role": "tool", "content": "x", "tool_call_id": "c1"},
        ]
    }
    assert span["output_data"] == {"content": "hello there"}
    sources = {seg["source"] for seg in span.get("context_segments") or []}
    assert "system_prompt" in sources


@pytest.mark.req("MHQ-1198")
def test_wrap_responses_tool_only_output_is_not_blank():
    response = _responses_response(text="", function_calls=[("lookup", '{"q": 1}')])
    client = av_openai.wrap(FakeClient(response))
    with tracing.run("agent"):
        client.responses.create(model="gpt-4o-mini", input="find it")

    assert _llm_spans()[0]["output_data"] == {"content": '[tool_call:lookup] {"q": 1}'}


@pytest.mark.req("MHQ-1198")
def test_wrap_responses_outside_trace_creates_auto_trace():
    client = av_openai.wrap(FakeClient())
    tracing._last_trace = None
    client.responses.create(model="gpt-4o-mini", input="hi")

    spans = _llm_spans()
    assert len(spans) == 1
    assert spans[0]["name"] == "openai.responses.create"


@pytest.mark.req("MHQ-1198")
def test_wrap_async_responses_create_is_traced():
    class _AsyncClient:
        def __init__(self) -> None:
            self.responses = FakeAsyncResponses(_responses_response())

    fake = _AsyncClient()
    type(fake).__name__ = "AsyncOpenAI"
    client = av_openai.wrap(fake)

    async def call() -> None:
        with tracing.run("agent"):
            await client.responses.create(model="gpt-4o-mini", input="hi")

    asyncio.run(call())
    spans = _llm_spans()
    assert len(spans) == 1
    assert spans[0]["input_tokens"] == 30


@pytest.mark.req("MHQ-1198")
def test_wrap_chat_still_traced_alongside_responses():
    chat_response = SimpleNamespace(
        model="gpt-4o-mini",
        usage=SimpleNamespace(
            prompt_tokens=5,
            completion_tokens=2,
            total_tokens=7,
            prompt_tokens_details=None,
            completion_tokens_details=None,
        ),
    )
    fake = FakeClient()
    fake.chat = SimpleNamespace(completions=FakeResponses(chat_response))
    client = av_openai.wrap(fake)
    with tracing.run("agent"):
        client.chat.completions.create(model="gpt-4o-mini", messages=[])
        client.responses.create(model="gpt-4o-mini", input="hi")

    names = sorted(s["name"] for s in _llm_spans())
    assert names == ["openai.chat.completions.create", "openai.responses.create"]


@pytest.fixture
def fake_openai_module(monkeypatch):
    """A minimal ``openai`` package exposing the two classes install() patches."""
    response = _responses_response()

    class Completions:
        def create(self, *args: Any, **kwargs: Any) -> Any:
            return response

    class AsyncCompletions:
        async def create(self, *args: Any, **kwargs: Any) -> Any:
            return response

    class Responses:
        def create(self, *args: Any, **kwargs: Any) -> Any:
            return response

    class AsyncResponses:
        async def create(self, *args: Any, **kwargs: Any) -> Any:
            return response

    openai_mod = ModuleType("openai")
    resources = ModuleType("openai.resources")
    chat = ModuleType("openai.resources.chat")
    completions = ModuleType("openai.resources.chat.completions")
    responses = ModuleType("openai.resources.responses")
    completions.Completions = Completions
    completions.AsyncCompletions = AsyncCompletions
    responses.Responses = Responses
    responses.AsyncResponses = AsyncResponses
    for name, mod in {
        "openai": openai_mod,
        "openai.resources": resources,
        "openai.resources.chat": chat,
        "openai.resources.chat.completions": completions,
        "openai.resources.responses": responses,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)
    return SimpleNamespace(Responses=Responses, original=Responses.create)


@pytest.mark.req("MHQ-1198")
def test_install_patches_and_uninstall_restores_responses(fake_openai_module):
    assert av_openai.install() is True
    try:
        with tracing.run("agent"):
            fake_openai_module.Responses().create(model="gpt-4o-mini", input="hi")
        spans = _llm_spans()
        assert [s["name"] for s in spans] == ["openai.responses.create"]
    finally:
        av_openai.uninstall()
    assert fake_openai_module.Responses.create is fake_openai_module.original


@pytest.mark.req("MHQ-1198")
def test_install_plus_wrap_does_not_double_count(fake_openai_module):
    av_openai.install()
    try:
        # wrap() is a no-op once install() is active.
        inner = SimpleNamespace(responses=fake_openai_module.Responses())
        client = av_openai.wrap(inner)
        with tracing.run("agent"):
            client.responses.create(model="gpt-4o-mini", input="hi")
        assert len(_llm_spans()) == 1
    finally:
        av_openai.uninstall()
