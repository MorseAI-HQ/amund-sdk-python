"""MHQ-1928: ``wrap()`` plus ``span(name, "tool")`` records each tool call once.

A raw client infers a ``tool`` span for every tool call it sees answered
(MHQ-1410). An app that also wraps the tool in ``amund.span(..., "tool")`` got
both, so every tool appeared twice. The manual span now stands in for the
inferred one and gains the ``tool_call_id``.
"""

from __future__ import annotations

from typing import Any

import pytest

openai = pytest.importorskip("openai")
httpx = pytest.importorskip("httpx")

import amund  # noqa: E402
from amund import tracing  # noqa: E402
from tests.test_launch_repro_openai import (  # noqa: E402
    _MODEL,
    _chat_body,
    _client,
    _spans,
    installed,  # noqa: F401  (fixture)
)

_CALLS = [
    {
        "id": "call_1",
        "type": "function",
        "function": {"name": "search_docs", "arguments": '{"q": "refund"}'},
    },
    {
        "id": "call_2",
        "type": "function",
        "function": {"name": "get_order", "arguments": '{"id": 7}'},
    },
]


def _inferred(span: dict[str, Any]) -> bool:
    return (span.get("metadata") or {}).get("timing") == "inferred"


def _loop(run_tool) -> None:  # noqa: ANN001
    bodies = iter(
        [_chat_body(content=None, tool_calls=_CALLS), _chat_body(content="Done")]
    )
    client = _client(lambda r: httpx.Response(200, json=next(bodies)))
    messages: list[dict[str, Any]] = [{"role": "user", "content": "refund 7"}]
    with tracing.run("support-bot"):
        first = client.chat.completions.create(model=_MODEL, messages=messages)
        messages.append(first.choices[0].message.model_dump(exclude_none=True))
        for call in first.choices[0].message.tool_calls:
            out = run_tool(call)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": out})
        client.chat.completions.create(model=_MODEL, messages=messages)


@pytest.mark.req("MHQ-1928")
def test_manual_tool_span_replaces_the_inferred_one(installed):  # noqa: F811
    def run_tool(call):  # noqa: ANN001, ANN202
        with amund.span(call.function.name, "tool", input_data={"q": 1}) as s:
            s.set_output("ok")
        return "ok"

    _loop(run_tool)

    tools = _spans("tool")
    assert sorted(t["name"] for t in tools) == ["get_order", "search_docs"]
    for t in tools:
        assert (t.get("metadata") or {}).get("timing") != "inferred"
        assert t["input_data"] == {"q": 1}
    by_name = {t["name"]: t for t in tools}
    assert by_name["search_docs"]["metadata"]["tool_call_id"] == "call_1"
    assert by_name["get_order"]["metadata"]["tool_call_id"] == "call_2"


@pytest.mark.req("MHQ-1928")
def test_manual_span_with_another_name_links_by_tool_call_id(installed):  # noqa: F811
    def run_tool(call):  # noqa: ANN001, ANN202
        with amund.span(f"my_{call.function.name}", "tool") as s:
            s.set_metadata("tool_call_id", call.id)
        return "ok"

    _loop(run_tool)

    names = sorted(t["name"] for t in _spans("tool"))
    assert names == ["my_get_order", "my_search_docs"]


@pytest.mark.req("MHQ-1928")
def test_tools_without_a_manual_span_are_still_inferred(installed):  # noqa: F811
    def run_tool(call):  # noqa: ANN001, ANN202
        if call.function.name == "search_docs":
            with amund.span("search_docs", "tool"):
                pass
        return "ok"

    _loop(run_tool)

    tools = {t["name"]: t for t in _spans("tool")}
    assert set(tools) == {"search_docs", "get_order"}
    assert (tools["search_docs"].get("metadata") or {}).get("timing") != "inferred"
    assert tools["get_order"]["metadata"]["timing"] == "inferred"


@pytest.mark.req("MHQ-1928")
def test_one_manual_span_never_stands_in_for_two_calls(installed):  # noqa: F811
    calls = [dict(c, function=dict(c["function"], name="lookup")) for c in _CALLS]
    bodies = iter(
        [_chat_body(content=None, tool_calls=calls), _chat_body(content="Done")]
    )
    client = _client(lambda r: httpx.Response(200, json=next(bodies)))
    messages: list[dict[str, Any]] = [{"role": "user", "content": "x"}]
    with tracing.run("support-bot"):
        first = client.chat.completions.create(model=_MODEL, messages=messages)
        messages.append(first.choices[0].message.model_dump(exclude_none=True))
        with amund.span("lookup", "tool"):  # only the first call is wrapped
            pass
        for call in first.choices[0].message.tool_calls:
            messages.append({"role": "tool", "tool_call_id": call.id, "content": "ok"})
        client.chat.completions.create(model=_MODEL, messages=messages)

    tools = _spans("tool")
    assert len(tools) == 2
    assert sum(_inferred(t) for t in tools) == 1


@pytest.mark.req("MHQ-1928")
def test_a_manual_span_from_before_the_call_is_not_matched(installed):  # noqa: F811
    bodies = iter(
        [_chat_body(content=None, tool_calls=_CALLS[:1]), _chat_body(content="Done")]
    )
    client = _client(lambda r: httpx.Response(200, json=next(bodies)))
    messages: list[dict[str, Any]] = [{"role": "user", "content": "x"}]
    with tracing.run("support-bot"):
        with amund.span("search_docs", "tool"):  # an earlier, unrelated run
            pass
        first = client.chat.completions.create(model=_MODEL, messages=messages)
        messages.append(first.choices[0].message.model_dump(exclude_none=True))
        messages.append({"role": "tool", "tool_call_id": "call_1", "content": "ok"})
        client.chat.completions.create(model=_MODEL, messages=messages)

    tools = _spans("tool")
    assert len(tools) == 2
    assert sum(_inferred(t) for t in tools) == 1
