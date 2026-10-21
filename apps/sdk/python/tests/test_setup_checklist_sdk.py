"""SETUP-09 / SETUP-10: Responses thread facts and opt-in hashed end-user capture.

Fakes only — no real API calls.
"""

from __future__ import annotations

import hashlib
from types import SimpleNamespace
from typing import Any

import pytest

import morse_ai
from morse_ai import tracing
from morse_ai.adapters import anthropic as av_anthropic
from morse_ai.adapters import openai as av_openai

RAW = "alice@x.com"
HASHED = "h:" + hashlib.sha256(RAW.encode()).hexdigest()[:32]


def _resp(rid: str | None = "r2") -> Any:
    return SimpleNamespace(
        id=rid,
        model="gpt-4o-mini",
        output_text="hi",
        output=[],
        usage=SimpleNamespace(
            input_tokens=3,
            output_tokens=2,
            total_tokens=5,
            input_tokens_details=None,
            output_tokens_details=None,
        ),
    )


def _chat_resp() -> Any:
    return SimpleNamespace(
        id="c1",
        model="gpt-4o-mini",
        choices=[],
        usage=SimpleNamespace(
            prompt_tokens=1,
            completion_tokens=1,
            total_tokens=2,
            prompt_tokens_details=None,
            completion_tokens_details=None,
        ),
    )


class _Responses:
    def create(self, *a: Any, **kw: Any) -> Any:
        return _resp()


class _Completions:
    def create(self, *a: Any, **kw: Any) -> Any:
        return _chat_resp()


class _OAI:
    def __init__(self) -> None:
        self.responses = _Responses()
        self.chat = SimpleNamespace(completions=_Completions())


class _Messages:
    def create(self, **kw: Any) -> Any:
        return SimpleNamespace(
            id="m1",
            model="claude-haiku-4-5",
            usage=SimpleNamespace(
                input_tokens=1,
                output_tokens=1,
                cache_creation_input_tokens=0,
                cache_read_input_tokens=0,
            ),
            content=[SimpleNamespace(type="text", text="x")],
        )


class _Anth:
    def __init__(self) -> None:
        self.messages = _Messages()


@pytest.fixture(autouse=True)
def _reset():
    from morse_ai.tracing import _active_trace

    _active_trace.set(None)
    morse_ai._reset_for_testing()
    av_openai.uninstall()
    tracing._set_capture_end_user(False)
    yield
    _active_trace.set(None)
    morse_ai._reset_for_testing()
    av_openai.uninstall()
    tracing._set_capture_end_user(False)


def _trace() -> dict[str, Any]:
    t = tracing._get_last_trace()
    assert t is not None
    return t


def _trace_fields() -> dict[str, Any]:
    return _trace()["trace"]


def _llm_span() -> dict[str, Any]:
    return [s for s in _trace()["spans"] if s.get("type") == "llm"][0]


@pytest.mark.req("SETUP-09")
def test_responses_call_carries_thread_facts():
    client = av_openai.wrap(_OAI())
    with tracing.run("agent"):
        client.responses.create(
            model="m", input="hi", previous_response_id="r1", conversation="conv_9"
        )
    md = _llm_span()["metadata"]
    assert md["response_id"] == "r2"
    assert md["previous_response_id"] == "r1"
    assert md["conversation_id"] == "conv_9"


@pytest.mark.req("SETUP-09")
def test_absent_keys_are_not_set():
    client = av_openai.wrap(_OAI())
    with tracing.run("agent"):
        client.responses.create(model="m", input="hi")
    assert _llm_span()["metadata"] == {"response_id": "r2"}


@pytest.mark.req("SETUP-09")
def test_conversation_object_and_dict_use_id():
    client = av_openai.wrap(_OAI())
    with tracing.run("agent"):
        client.responses.create(
            model="m", input="hi", conversation=SimpleNamespace(id="conv_obj")
        )
    assert _llm_span()["metadata"]["conversation_id"] == "conv_obj"
    with tracing.run("agent"):
        client.responses.create(model="m", input="hi", conversation={"id": "conv_d"})
    assert _llm_span()["metadata"]["conversation_id"] == "conv_d"


@pytest.mark.req("SETUP-09")
def test_chat_completions_gets_no_thread_facts():
    client = av_openai.wrap(_OAI())
    with tracing.run("agent"):
        client.chat.completions.create(model="m", messages=[])
    assert not _llm_span().get("metadata")


@pytest.mark.req("SETUP-09")
def test_thread_facts_reach_the_otlp_wire_attributes():
    from morse_ai.otel_provider import _span_dict_to_otel_attributes

    client = av_openai.wrap(_OAI())
    with tracing.run("agent"):
        client.responses.create(model="m", input="hi", previous_response_id="r1")
    attrs = _span_dict_to_otel_attributes(
        _llm_span(), _trace(), {}, is_root=False, max_field_bytes=65536
    )
    assert attrs["morse.metadata.response_id"] == '"r2"'
    assert attrs["morse.metadata.previous_response_id"] == '"r1"'


@pytest.mark.req("SETUP-10")
def test_end_user_off_by_default_never_sent():
    client = av_openai.wrap(_OAI())
    with tracing.run("agent"):
        client.responses.create(model="m", input="hi", user=RAW)
    t = _trace()
    assert "user_id" not in t["trace"]
    assert RAW not in repr(t)


@pytest.mark.req("SETUP-10")
def test_init_option_defaults_off_and_enables():
    morse_ai.init("mhq_test_key")
    assert tracing._capture_end_user is False
    morse_ai.init("mhq_test_key", capture_end_user=True)
    assert tracing._capture_end_user is True


@pytest.mark.req("SETUP-10")
def test_openai_user_hashed_when_on():
    tracing._set_capture_end_user(True)
    client = av_openai.wrap(_OAI())
    with tracing.run("agent"):
        client.responses.create(model="m", input="hi", user=RAW)
    t = _trace()
    assert t["trace"]["user_id"] == HASHED
    assert RAW not in repr(t)


@pytest.mark.req("SETUP-10")
def test_openai_safety_identifier_and_chat_path():
    tracing._set_capture_end_user(True)
    client = av_openai.wrap(_OAI())
    with tracing.run("agent"):
        client.responses.create(model="m", input="hi", safety_identifier=RAW)
    assert _trace_fields()["user_id"] == HASHED
    with tracing.run("agent"):
        client.chat.completions.create(model="m", messages=[], user=RAW)
    assert _trace_fields()["user_id"] == HASHED


@pytest.mark.req("SETUP-10")
def test_app_set_user_id_wins_and_customer_id_never_set():
    tracing._set_capture_end_user(True)
    client = av_openai.wrap(_OAI())
    with tracing.run("agent", user_id="app-user-1"):
        client.responses.create(model="m", input="hi", user=RAW)
    assert _trace_fields()["user_id"] == "app-user-1"
    with tracing.run("agent"):
        client.responses.create(model="m", input="hi", user=RAW)
    t = _trace_fields()
    assert t["user_id"] == HASHED
    assert "customer_id" not in t
    assert "customer_id" not in (t.get("user_metadata") or {})


@pytest.mark.req("SETUP-10")
def test_anthropic_metadata_user_id():
    client = av_anthropic.wrap(_Anth())
    kwargs: dict[str, Any] = {
        "model": "claude-haiku-4-5",
        "max_tokens": 5,
        "messages": [{"role": "user", "content": "hi"}],
        "metadata": {"user_id": RAW},
    }
    with tracing.run("agent"):
        client.messages.create(**kwargs)
    assert "user_id" not in _trace_fields()  # off by default

    tracing._set_capture_end_user(True)
    with tracing.run("agent"):
        client.messages.create(**kwargs)
    t = _trace()
    assert t["trace"]["user_id"] == HASHED
    assert RAW not in repr(t)
