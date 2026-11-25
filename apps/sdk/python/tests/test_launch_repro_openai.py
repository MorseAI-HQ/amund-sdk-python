"""Launch-verification repros for the raw OpenAI adapter (WP-3, 2026-10-24).

Real ``openai`` SDK, HTTP mocked with ``httpx.MockTransport``. No key, no
network. Response bodies follow the shapes recorded in the 2026-10-18 dogfood
round (``bizops/dogfood/ROUND-2026-10-18-sdk-0.7.0.md``).

- MHQ-1236: ``cached_input_tokens`` reaches the span (0 when nothing cached).
- MHQ-1407: ``stream=True`` on Chat Completions and Responses records model,
  tokens and cost when the stream ends (was model/tokens/cost NULL, $0).
- MHQ-1410: a failed call leaves a failed ``llm`` span; a tool loop yields
  ``tool`` spans; the Responses tool loop's input keeps the function items.
- MHQ-1506: the auto-installed raw client does not double-count an
  openai-agents run.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

openai = pytest.importorskip("openai")
httpx = pytest.importorskip("httpx")

from amund import tracing  # noqa: E402
from amund.adapters import openai as av_openai  # noqa: E402

_MODEL = "gpt-4.1-nano-2025-04-14"
_BASE = "https://api.openai.test/v1"


def _usage(cached: int = 0) -> dict[str, Any]:
    return {
        "prompt_tokens": 120,
        "completion_tokens": 8,
        "total_tokens": 128,
        "prompt_tokens_details": {"cached_tokens": cached},
        "completion_tokens_details": {"reasoning_tokens": 0},
    }


def _chat_body(
    *, content: str | None = "Hello", tool_calls: list | None = None, cached: int = 0
) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": _MODEL,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": "tool_calls" if tool_calls else "stop",
            }
        ],
        "usage": _usage(cached),
    }


def _sse(events: list[dict[str, Any]], *, done: bool = True) -> bytes:
    out = b"".join(b"data: " + json.dumps(e).encode() + b"\n\n" for e in events)
    return out + (b"data: [DONE]\n\n" if done else b"")


def _chat_chunks(*, include_usage: bool) -> list[dict[str, Any]]:
    base = {"id": "chatcmpl-s", "object": "chat.completion.chunk", "created": 0}
    chunks = [
        {
            **base,
            "model": _MODEL,
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Hel"}}],
        },
        {
            **base,
            "model": _MODEL,
            "choices": [{"index": 0, "delta": {"content": "lo"}}],
        },
        {
            **base,
            "model": _MODEL,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        },
    ]
    if include_usage:
        chunks.append({**base, "model": _MODEL, "choices": [], "usage": _usage(16)})
    return chunks


def _response_obj(status: str = "completed") -> dict[str, Any]:
    return {
        "id": "resp_1",
        "object": "response",
        "created_at": 0,
        "model": _MODEL,
        "status": status,
        "output": [
            {
                "type": "message",
                "id": "msg_1",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Hi!", "annotations": []}],
            }
        ],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "usage": {
            "input_tokens": 50,
            "output_tokens": 4,
            "total_tokens": 54,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
    }


def _responses_events() -> list[dict[str, Any]]:
    created = {**_response_obj("in_progress"), "output": [], "usage": None}
    return [
        {"type": "response.created", "sequence_number": 0, "response": created},
        {
            "type": "response.output_text.delta",
            "sequence_number": 1,
            "item_id": "msg_1",
            "output_index": 0,
            "content_index": 0,
            "delta": "Hi!",
            "logprobs": [],
        },
        {
            "type": "response.completed",
            "sequence_number": 2,
            "response": _response_obj(),
        },
    ]


def _client(handler, *, is_async: bool = False):  # noqa: ANN001, ANN202
    kw = {
        "api_key": "sk-test-not-a-real-key",  # pragma: allowlist secret
        "base_url": _BASE,
        "max_retries": 0,
    }
    if is_async:
        return openai.AsyncOpenAI(
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)), **kw
        )
    return openai.OpenAI(
        http_client=httpx.Client(transport=httpx.MockTransport(handler)), **kw
    )


def _stream_response(events: list[dict[str, Any]], **kw: Any):  # noqa: ANN202
    return httpx.Response(
        200, headers={"content-type": "text/event-stream"}, content=_sse(events, **kw)
    )


@pytest.fixture
def installed():
    token = tracing._active_trace.set(None)
    av_openai.uninstall()
    av_openai.install()
    try:
        yield
    finally:
        av_openai.uninstall()
        tracing._active_trace.reset(token)


def _spans(kind: str) -> list[dict[str, Any]]:
    trace = tracing._get_last_trace()
    assert trace is not None
    return [s for s in trace["spans"] if s["type"] == kind]


# ---------------------------------------------------------------------------
# MHQ-1236
# ---------------------------------------------------------------------------


@pytest.mark.req("MHQ-1236")
@pytest.mark.parametrize("cached", [0, 96])
def test_mhq1236_cached_input_tokens_reach_the_span(installed, cached):
    client = _client(lambda r: httpx.Response(200, json=_chat_body(cached=cached)))
    with tracing.run("cache"):
        client.chat.completions.create(
            model=_MODEL, messages=[{"role": "user", "content": "hi"}]
        )
    (llm,) = _spans("llm")
    assert llm["cached_input_tokens"] == cached


# ---------------------------------------------------------------------------
# MHQ-1407 — streaming
# ---------------------------------------------------------------------------


@pytest.mark.req("MHQ-1407")
def test_mhq1407_chat_stream_with_usage_records_model_tokens_cost(installed):
    client = _client(lambda r: _stream_response(_chat_chunks(include_usage=True)))
    with tracing.run("stream-chat"):
        stream = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
            stream_options={"include_usage": True},
        )
        text = "".join(c.choices[0].delta.content or "" for c in stream if c.choices)
    assert text == "Hello"
    (llm,) = _spans("llm")
    assert llm["model"] == _MODEL
    assert llm["input_tokens"] == 120
    assert llm["output_tokens"] == 8
    assert llm["cached_input_tokens"] == 16
    assert llm["cost_usd"] and llm["cost_usd"] > 0
    assert llm["output_data"] == {"content": "Hello"}
    assert llm["metadata"]["streamed"] is True


@pytest.mark.req("MHQ-1407")
def test_mhq1407_chat_stream_without_usage_keeps_model_and_says_so(installed):
    client = _client(lambda r: _stream_response(_chat_chunks(include_usage=False)))
    with tracing.run("stream-chat-nousage"):
        stream = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
        )
        # LiteLLM-style consumption: next() rather than a for loop.
        while True:
            try:
                next(stream)
            except StopIteration:
                break
    (llm,) = _spans("llm")
    assert llm["model"] == _MODEL
    assert llm.get("input_tokens") is None
    assert llm["metadata"]["usage_unavailable"] is True


@pytest.mark.req("MHQ-1407")
def test_mhq1407_chat_stream_helper_is_traced(installed):
    client = _client(lambda r: _stream_response(_chat_chunks(include_usage=True)))
    with (
        tracing.run("stream-helper"),
        client.chat.completions.stream(
            model="gpt-4.1-nano",
            messages=[{"role": "user", "content": "hi"}],
            stream_options={"include_usage": True},
        ) as stream,
    ):
        final = stream.get_final_completion()
    assert final.choices[0].message.content == "Hello"
    (llm,) = _spans("llm")
    assert llm["input_tokens"] == 120


@pytest.mark.req("MHQ-1407")
def test_mhq1407_responses_stream_records_usage(installed):
    client = _client(lambda r: _stream_response(_responses_events(), done=False))
    with tracing.run("stream-responses"):
        for _event in client.responses.create(
            model="gpt-4.1-nano", input="hi", stream=True
        ):
            pass
    (llm,) = _spans("llm")
    assert llm["name"] == "openai.responses.create"
    assert llm["model"] == _MODEL
    assert llm["input_tokens"] == 50
    assert llm["output_tokens"] == 4
    assert llm["cost_usd"] and llm["cost_usd"] > 0


@pytest.mark.req("MHQ-1407")
def test_mhq1407_async_wrapped_stream_records_usage():
    client = av_openai.wrap(
        _client(
            lambda r: _stream_response(_chat_chunks(include_usage=True)), is_async=True
        )
    )

    async def main() -> None:
        with tracing.run("stream-async"):
            stream = await client.chat.completions.create(
                model="gpt-4.1-nano",
                messages=[{"role": "user", "content": "hi"}],
                stream=True,
                stream_options={"include_usage": True},
            )
            async for _chunk in stream:
                pass

    token = tracing._active_trace.set(None)
    try:
        asyncio.run(main())
    finally:
        tracing._active_trace.reset(token)
    (llm,) = _spans("llm")
    assert llm["output_tokens"] == 8


@pytest.mark.req("MHQ-1407")
def test_mhq1407_stream_without_a_trace_still_lands_in_one(installed):
    client = _client(lambda r: _stream_response(_chat_chunks(include_usage=True)))
    stream = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        stream_options={"include_usage": True},
    )
    for _chunk in stream:
        pass
    (llm,) = _spans("llm")
    assert llm["input_tokens"] == 120


@pytest.mark.req("MHQ-1407", "MHQ-1506")
def test_mhq1407_raw_response_path_used_by_litellm(installed):
    """LiteLLM's OpenAI provider calls ``with_raw_response.create``."""
    client = _client(lambda r: httpx.Response(200, json=_chat_body()))
    with tracing.run("raw"):
        raw = client.chat.completions.with_raw_response.create(
            model="gpt-4.1-nano", messages=[{"role": "user", "content": "hi"}]
        )
        parsed = raw.parse()
    assert parsed.choices[0].message.content == "Hello"
    (llm,) = _spans("llm")
    assert llm["model"] == _MODEL
    assert llm["input_tokens"] == 120


@pytest.mark.req("MHQ-1407")
def test_mhq1407_raw_response_stream_path_used_by_litellm(installed):
    client = _client(lambda r: _stream_response(_chat_chunks(include_usage=True)))
    with tracing.run("raw-stream"):
        raw = client.chat.completions.with_raw_response.create(
            model="gpt-4.1-nano",
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
            stream_options={"include_usage": True},
        )
        for _chunk in raw.parse():
            pass
    (llm,) = _spans("llm")
    assert llm["input_tokens"] == 120


# ---------------------------------------------------------------------------
# MHQ-1410
# ---------------------------------------------------------------------------


@pytest.mark.req("MHQ-1410")
def test_mhq1410_failed_call_leaves_a_failed_llm_span(installed):
    def handler(request):  # noqa: ANN001, ANN202
        return httpx.Response(
            404,
            json={
                "error": {
                    "message": "The model `gpt-nope` does not exist",
                    "type": "invalid_request_error",
                    "code": "model_not_found",
                }
            },
        )

    client = _client(handler)
    with pytest.raises(openai.NotFoundError), tracing.run("fail"):
        client.chat.completions.create(
            model="gpt-nope", messages=[{"role": "user", "content": "hi"}]
        )
    (llm,) = _spans("llm")
    assert llm["status"] == "failed"
    assert llm["model"] == "gpt-nope"
    assert "does not exist" in llm["error_message"]
    assert llm["exception_type"].endswith("NotFoundError")
    assert llm["duration_ms"] is not None
    assert not llm.get("cost_usd")


@pytest.mark.req("MHQ-1410")
def test_mhq1410_chat_tool_loop_emits_tool_spans(installed):
    """openai-cookbook routines/handoffs shape: two tool calls, then text."""
    tool_calls = [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "execute_order", "arguments": '{"item": 7}'},
        },
        {
            "id": "call_2",
            "type": "function",
            "function": {"name": "transfer_to_sales_agent", "arguments": "{}"},
        },
    ]
    bodies = iter(
        [_chat_body(content=None, tool_calls=tool_calls), _chat_body(content="Done")]
    )
    client = _client(lambda r: httpx.Response(200, json=next(bodies)))
    messages: list[dict[str, Any]] = [{"role": "user", "content": "buy 7"}]
    with tracing.run("routines"):
        first = client.chat.completions.create(model=_MODEL, messages=messages)
        messages.append(first.choices[0].message.model_dump(exclude_none=True))
        messages.append({"role": "tool", "tool_call_id": "call_1", "content": "ok"})
        messages.append(
            {"role": "tool", "tool_call_id": "call_2", "content": "Sales Agent"}
        )
        client.chat.completions.create(model=_MODEL, messages=messages)

    tools = {s["name"]: s for s in _spans("tool")}
    assert set(tools) == {"execute_order", "transfer_to_sales_agent"}
    assert tools["execute_order"]["input_data"] == {"arguments": '{"item": 7}'}
    assert tools["execute_order"]["output_data"] == {"content": "ok"}
    assert tools["execute_order"]["metadata"]["timing"] == "inferred"
    assert tools["transfer_to_sales_agent"]["metadata"]["handoff_target"] == (
        "sales_agent"
    )
    llms = _spans("llm")
    assert len(llms) == 2
    # The second call's input shows the assistant tool calls, not a blank turn.
    second_input = llms[1]["input_data"]["messages"]
    assert second_input[1]["tool_calls"] == [
        '[tool_call:execute_order] {"item": 7}',
        "[tool_call:transfer_to_sales_agent] {}",
    ]


@pytest.mark.req("MHQ-1410")
def test_mhq1410_responses_tool_loop_emits_tool_span_and_keeps_input_items(
    installed,
):
    call = {
        "type": "function_call",
        "id": "fc_1",
        "call_id": "call_9",
        "name": "get_weather",
        "arguments": '{"city": "Paris"}',
        "status": "completed",
    }
    bodies = iter([{**_response_obj(), "output": [call]}, _response_obj()])
    client = _client(lambda r: httpx.Response(200, json=next(bodies)))
    with tracing.run("responses-tools"):
        first = client.responses.create(model=_MODEL, input="weather?")
        client.responses.create(
            model=_MODEL,
            input=[
                {"role": "user", "content": "weather?"},
                *[item.model_dump() for item in first.output],
                {"type": "function_call_output", "call_id": "call_9", "output": "18C"},
            ],
        )
    (tool,) = _spans("tool")
    assert tool["name"] == "get_weather"
    assert tool["output_data"] == {"content": "18C"}
    second_input = _spans("llm")[1]["input_data"]["messages"]
    assert [m["role"] for m in second_input] == ["user", "assistant", "tool"]


# ---------------------------------------------------------------------------
# MHQ-1506 — the raw client patch does not double-count framework calls
# ---------------------------------------------------------------------------


@pytest.mark.req("MHQ-1506")
def test_mhq1506_openai_agents_run_is_not_double_counted(installed):
    agents = pytest.importorskip("agents")
    from amund.adapters import openai_agents as av_agents

    def handler(request):  # noqa: ANN001, ANN202
        return httpx.Response(200, json=_response_obj())

    av_agents.uninstall()
    av_agents.install()
    agents.set_tracing_disabled(True)
    agents.set_default_openai_api("responses")
    agents.set_default_openai_client(
        _client(handler, is_async=True), use_for_tracing=False
    )
    try:
        agent = agents.Agent(name="Assistant", model="gpt-4.1-nano")
        result = agents.Runner.run_sync(agent, "hi")
        assert result.final_output == "Hi!"
    finally:
        av_agents.uninstall()
    llm = _spans("llm")
    assert len(llm) == 1, [s["name"] for s in llm]
    assert llm[0]["name"] == "openai_agents.llm"
