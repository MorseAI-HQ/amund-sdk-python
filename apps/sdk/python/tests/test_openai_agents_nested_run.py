"""MHQ-1154 — OpenAI Agents spans must survive `with morse_ai.run(...)` nesting.

Adapter: `apps/sdk/python/src/morse_ai/adapters/openai_agents.py`
Root cause: `apps/sdk/python/src/morse_ai/_internal/auto_detect.py`

The reported driver called ``morse_ai.init()`` BEFORE ``from agents import
Runner``. That path relies on the lazy import hook, which installed the adapter
from a background thread that only got to patch ``Runner`` after the import
statement returned — so the very next ``Runner.run_sync`` ran unpatched and
the trace held only the caller's ``run()`` span. The "bare" driver that looked
fine imported ``agents`` first and took the synchronous install path. Nesting
was never the variable; import order was.

Each scenario runs in a fresh interpreter: the bug only exists while
``agents`` has not been imported yet, which can't be recreated in-process.
Real ``Runner`` + a fake ``Model`` — no network.
"""

# ruff: noqa: E501
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

import pytest

pytest.importorskip("agents", reason="openai-agents not installed")

_PRELUDE = textwrap.dedent(
    """
    import asyncio, json
    from morse_ai._internal import auto_detect
    from morse_ai import tracing
    import morse_ai
    from openai.types.responses import (
        ResponseFunctionToolCall, ResponseOutputMessage, ResponseOutputText,
    )

    if IMPORT_ORDER == "init_first":
        auto_detect.install_adapters()  # what morse_ai.init() does

    # One import statement, then straight to work -- the shape of the
    # reported driver. Anything slow in between hides the race.
    from agents import (
        Agent, Model, ModelResponse, Runner, Usage, function_tool,
        set_tracing_disabled,
    )
    if IMPORT_ORDER == "agents_first":
        auto_detect.install_adapters()  # sys.modules scan path
    # Read before ANY other work: the old background-thread install patched
    # Runner a few ms after this line, so whether the traced call won the race
    # depended on scheduling (it lost on dev with a live transport).
    PATCHED_AT_IMPORT = hasattr(Runner.run_sync, "__wrapped__")

    set_tracing_disabled(True)

    @function_tool
    def add(a: int, b: int) -> int:
        "Add two ints"
        return a + b

    class FakeModel(Model):
        def __init__(self):
            self.n = 0
        async def get_response(self, *a, **k):
            self.n += 1
            if self.n == 1:
                out = [ResponseFunctionToolCall(
                    type="function_call", call_id="c1", name="add",
                    arguments='{"a": 2, "b": 3}', id="fc1")]
            else:
                out = [ResponseOutputMessage(
                    id="m1", type="message", role="assistant", status="completed",
                    content=[ResponseOutputText(type="output_text", text="5", annotations=[])])]
            return ModelResponse(
                output=out,
                usage=Usage(requests=1, input_tokens=10, output_tokens=5, total_tokens=15),
                response_id=None)
        def stream_response(self, *a, **k):
            raise NotImplementedError

    agent = Agent(name="Calc", model=FakeModel(), tools=[add])
    """
)

_SCENARIOS = {
    "run_sync_nested": """
with morse_ai.run("agent-x") as r:
    Runner.run_sync(agent, "2+3?")
    r.set_outcome(True, "ok")
""",
    "run_async_nested": """
async def main():
    with morse_ai.run("agent-x") as r:
        await Runner.run(agent, "2+3?")
        r.set_outcome(True, "ok")
asyncio.run(main())
""",
    "run_sync_standalone": """
Runner.run_sync(agent, "2+3?")
""",
}


def _run(scenario: str, import_order: str) -> dict:
    code = (
        f"IMPORT_ORDER = {import_order!r}\n"
        + _PRELUDE
        + _SCENARIOS[scenario]
        + "\nt = tracing._get_last_trace()\n"
        + "print('@@' + json.dumps({'patched': PATCHED_AT_IMPORT, 'trace': t['trace'], 'spans': [{k: s.get(k) "
        + "for k in ('span_id', 'trace_id', 'parent_span_id', 'type', 'name')} "
        + "for s in t['spans']]}))\n"
    )
    env = {k: v for k, v in os.environ.items() if not k.startswith("MORSE_")}
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    line = next(ln for ln in proc.stdout.splitlines() if ln.startswith("@@"))
    return json.loads(line[2:])


def _assert_full_trace(result: dict, root_name: str) -> None:
    assert result["patched"], "Runner was not instrumented when the import returned"
    spans = result["spans"]
    kinds = [s["type"] for s in spans]
    assert kinds.count("llm") == 2, f"LLM spans dropped: {kinds}"
    assert kinds.count("tool") == 1, f"tool span dropped: {kinds}"
    trace_id = result["trace"]["trace_id"]
    assert {s["trace_id"] for s in spans} == {trace_id}
    roots = [s for s in spans if not s["parent_span_id"]]
    assert len(roots) == 1 and roots[0]["name"] == root_name
    for s in spans:
        if s is not roots[0]:
            assert s["parent_span_id"] == roots[0]["span_id"]


@pytest.mark.req("MHQ-1154")
@pytest.mark.parametrize("import_order", ["init_first", "agents_first"])
def test_run_sync_nested_in_morse_run_keeps_all_spans(import_order):
    _assert_full_trace(_run("run_sync_nested", import_order), "agent-x")


@pytest.mark.req("MHQ-1154")
@pytest.mark.parametrize("import_order", ["init_first", "agents_first"])
def test_async_run_nested_in_morse_run_keeps_all_spans(import_order):
    _assert_full_trace(_run("run_async_nested", import_order), "agent-x")


@pytest.mark.req("MHQ-1154")
@pytest.mark.parametrize("import_order", ["init_first", "agents_first"])
def test_run_sync_standalone_still_traced(import_order):
    _assert_full_trace(_run("run_sync_standalone", import_order), "Calc")
