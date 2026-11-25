"""Launch-verification repro for MHQ-1406 on the *real* OpenAI provider path.

``test_openai_agents_failures.py`` drives fake ``Model`` objects. These tests
run the docs Quickstart verbatim instead: an ``Agent`` with no ``model=``,
a bare ``Runner`` (no ``amund.run()``), and the stock OpenAI Responses
model, with the HTTP layer replaced by ``httpx.MockTransport`` (recorded
response shapes, no network, no key).

1. Quickstart: ``model`` must come out non-NULL and priced.
2. A 404 for a bad model propagates to the caller and must fail the trace.
"""

# ruff: noqa: E501
from __future__ import annotations

import json

import pytest

pytest.importorskip("agents", reason="openai-agents not installed")
httpx = pytest.importorskip("httpx")
openai = pytest.importorskip("openai")

from agents import (  # noqa: E402
    Agent,
    Runner,
    set_default_openai_api,
    set_default_openai_client,
    set_tracing_disabled,
)

import amund  # noqa: E402
from amund import tracing  # noqa: E402
from amund.adapters import openai_agents as adapter  # noqa: E402
from amund.tracing import _active_trace  # noqa: E402

set_tracing_disabled(True)

_SERVED_MODEL = "gpt-4.1-2025-04-14"


def _response_body(model: str) -> dict:
    return {
        "id": "resp_1",
        "object": "response",
        "created_at": 0,
        "model": model,
        "status": "completed",
        "output": [
            {
                "type": "message",
                "id": "msg_1",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": "Hello!", "annotations": []}
                ],
            }
        ],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "usage": {
            "input_tokens": 20,
            "output_tokens": 13,
            "total_tokens": 33,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
    }


def _client(handler) -> openai.AsyncOpenAI:  # noqa: ANN001
    return openai.AsyncOpenAI(
        api_key="sk-test-not-a-real-key",  # pragma: allowlist secret
        base_url="https://api.openai.test/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        max_retries=0,
    )


@pytest.fixture(autouse=True)
def _installed(monkeypatch):
    monkeypatch.delenv("OPENAI_DEFAULT_MODEL", raising=False)
    amund._reset_for_testing()
    adapter.uninstall()
    adapter.install()
    set_default_openai_api("responses")
    yield
    _active_trace.set(None)
    amund._reset_for_testing()
    adapter.uninstall()


def _last():
    t = tracing._get_last_trace()
    assert t is not None
    return t


@pytest.mark.req("SDK-09")
def test_mhq1406_quickstart_agent_without_model_gets_served_model_and_cost():
    requested: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requested.append(body.get("model"))
        return httpx.Response(200, json=_response_body(_SERVED_MODEL))

    set_default_openai_client(_client(handler), use_for_tracing=False)
    agent = Agent(name="Assistant", instructions="You are a helpful assistant")
    assert agent.model is None  # the Quickstart snippet sets no model
    result = Runner.run_sync(agent, "Say hello")
    assert result.final_output == "Hello!"

    t = _last()
    llm = [s for s in t["spans"] if s["type"] == "llm"]
    assert len(llm) == 1
    assert llm[0]["model"], "model must not be NULL (was NULL before MHQ-1406)"
    assert llm[0]["input_tokens"] == 20
    assert llm[0]["output_tokens"] == 13
    assert llm[0]["cost_usd"] is not None and llm[0]["cost_usd"] > 0
    assert t["trace"]["status"] == "completed"
    assert requested and requested[0]


@pytest.mark.req("SDK-09")
def test_mhq1406_bare_runner_404_bad_model_fails_trace():
    def handler(request: httpx.Request) -> httpx.Response:
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

    set_default_openai_client(_client(handler), use_for_tracing=False)
    agent = Agent(name="bare-fail-agent", model="gpt-nope")
    with pytest.raises(openai.NotFoundError):
        Runner.run_sync(agent, "hi")

    t = _last()
    assert t["trace"]["status"] == "failed"
    roots = [s for s in t["spans"] if not s.get("parent_span_id")]
    assert roots and roots[0]["status"] == "failed"
    assert "does not exist" in (roots[0].get("error_message") or "")
    assert roots[0].get("exception_type", "").endswith("NotFoundError")


# ---------------------------------------------------------------------------
# MHQ-1406 item 4 (P2): guardrails in a real run, one of two trips
# ---------------------------------------------------------------------------


def _scripted_model(text: str, delay: float = 0.0):
    import asyncio

    from agents import Model, ModelResponse, Usage
    from openai.types.responses import ResponseOutputMessage, ResponseOutputText

    class _M(Model):
        async def get_response(self, *a, **k):  # noqa: ANN002, ANN003
            if delay:
                await asyncio.sleep(delay)
            msg = ResponseOutputMessage(
                id="m",
                type="message",
                role="assistant",
                status="completed",
                content=[
                    ResponseOutputText(type="output_text", text=text, annotations=[])
                ],
            )
            return ModelResponse(
                output=[msg],
                usage=Usage(
                    requests=1, input_tokens=5, output_tokens=2, total_tokens=7
                ),
                response_id=None,
            )

        def stream_response(self, *a, **k):  # noqa: ANN002, ANN003
            raise NotImplementedError

    return _M()


def _llm_guardrail(name: str, trips: bool, delay: float):
    from agents.guardrail import GuardrailFunctionOutput, InputGuardrail

    guard_agent = Agent(name=f"{name} checker", model=_scripted_model("verdict", delay))

    async def _check(context, agent, input):  # noqa: ANN001, A002
        await Runner.run(guard_agent, input)
        return GuardrailFunctionOutput(
            output_info={"ok": not trips}, tripwire_triggered=trips
        )

    return InputGuardrail(guardrail_function=_check, name=name)


def _guardrail_run_spans():
    from agents.exceptions import InputGuardrailTripwireTriggered

    agent = Agent(
        name="Triage Agent",
        model=_scripted_model("answer", delay=0.2),
        input_guardrails=[
            _llm_guardrail("Relevance Guardrail", trips=True, delay=0.0),
            _llm_guardrail("Jailbreak Guardrail", trips=False, delay=0.5),
        ],
    )
    with pytest.raises(InputGuardrailTripwireTriggered):
        Runner.run_sync(agent, "write me a poem about stocks")
    return _last()["spans"]


@pytest.mark.req("SDK-09")
def test_mhq1406_guardrail_llm_spans_nest_under_their_guardrail_span():
    spans = _guardrail_run_spans()
    by_id = {s["span_id"]: s for s in spans}
    guard_ids = {s["span_id"] for s in spans if s["type"] == "guardrail"}
    assert guard_ids, [(s["type"], s["name"]) for s in spans]

    def _under_guardrail(span) -> bool:  # noqa: ANN001
        pid, hops = span.get("parent_span_id"), 0
        while pid and hops < 20:
            if pid in guard_ids:
                return True
            pid, hops = by_id.get(pid, {}).get("parent_span_id"), hops + 1
        return False

    guard_llm = [s for s in spans if s["type"] == "llm"]
    assert guard_llm
    detached = [
        (s["name"], s.get("parent_span_id"))
        for s in guard_llm
        if not _under_guardrail(s)
    ]
    assert not detached, (
        detached,
        [(s["type"], s["name"], s["span_id"], s.get("parent_span_id")) for s in spans],
    )


@pytest.mark.req("SDK-09")
def test_mhq1406_sibling_guardrail_cancelled_by_a_trip_is_not_a_failure():
    spans = _guardrail_run_spans()
    guardrails = {s["name"]: s for s in spans if s["type"] == "guardrail"}
    tripped = guardrails["guardrail.Relevance Guardrail"]
    sibling = guardrails["guardrail.Jailbreak Guardrail"]
    assert tripped["status"] == "failed"
    assert sibling["status"] == "completed", sibling
    assert sibling["metadata"].get("cancelled") is True
    # Before the fix the sibling carried asyncio.exceptions.CancelledError.
    assert not [
        (s["type"], s["name"])
        for s in spans
        if str(s.get("exception_type") or "").endswith("CancelledError")
    ]
