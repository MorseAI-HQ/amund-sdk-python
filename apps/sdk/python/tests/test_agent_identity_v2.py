"""agent-identity-v2 (MHQ-1408), Python SDK wave: AID-01, 02, 04, 05, 09.

Graphs are real (tiny) LangGraph apps on a fake chat model, shaped like the
2026-10-18 dogfood recordings (bizops/dogfood/rounds/2026-10-18/group-B, -D):

* async LangGraph with no ``run()`` captured 2 spans (MHQ-1402, group B F1)
* the 14-analyst hedge-fund fan-out was 218 spans / 176 agent-typed (group D)
* LCEL prompt|model|parser nested flat, parallel branches cross-parented
"""

# ruff: noqa: E501
from __future__ import annotations

import asyncio
import operator
import sys
from types import SimpleNamespace
from typing import Annotated, Any, TypedDict

import pytest

from amund import tracing
from amund._internal.redaction import RedactionConfig
from amund.adapters.langgraph import LangGraphCallbackHandler
from amund.otel_provider import get_engine

pytest.importorskip("langgraph")
pytest.importorskip("langchain_core")

from langchain_core.language_models.chat_models import BaseChatModel  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402
from langchain_core.output_parsers import StrOutputParser  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402
from langchain_core.prompts import ChatPromptTemplate  # noqa: E402
from langgraph.graph import END, START, StateGraph  # noqa: E402

ANALYSTS = [f"analyst_{i:02d}" for i in range(14)]


class _FakeChat(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "fake"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):  # noqa: ANN001
        msg = AIMessage(
            content="hi",
            usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            response_metadata={"model_name": "gpt-4o-mini"},
        )
        return ChatResult(generations=[ChatGeneration(message=msg)])


_chain = ChatPromptTemplate.from_template("{x}") | _FakeChat() | StrOutputParser()


class _State(TypedDict):
    x: str
    out: Annotated[list, operator.add]


def _fanout_graph(name: str | None, analysts: list[str], async_nodes: bool = True):
    """start -> N analysts (each an LCEL prompt|model|parser call) -> manager."""

    def make(n: str):
        if async_nodes:

            async def node(s):  # noqa: ANN001
                await _chain.ainvoke({"x": n})
                return {"out": [n]}

        else:

            def node(s):  # noqa: ANN001
                _chain.invoke({"x": n})
                return {"out": [n]}

        return node

    def manager(s):  # noqa: ANN001  -- no LLM: framework bookkeeping, not an agent
        return {"out": ["done"]}

    g = StateGraph(_State)
    for n in analysts:
        g.add_node(n, make(n))
        g.add_edge(START, n)
        g.add_edge(n, "manager")
    g.add_node("manager", manager)
    g.add_edge("manager", END)
    return g.compile(name=name) if name else g.compile()


def _react_graph(name: str):
    """Prebuilt-ReAct shape: ``agent`` (LLM) <-> ``tools`` nodes."""

    async def agent(s):  # noqa: ANN001
        await _chain.ainvoke({"x": "q"})
        return {"out": ["a"]}

    def tools(s):  # noqa: ANN001
        return {"out": ["t"]}

    g = StateGraph(_State)
    g.add_node("agent", agent)
    g.add_node("tools", tools)
    g.add_edge(START, "agent")
    g.add_edge("agent", "tools")
    g.add_edge("tools", END)
    return g.compile(name=name)


def _invoke(graph, *, run_name: str | None = ..., handler: Any = True):  # type: ignore[assignment]
    """Run ``graph`` async, with ``run(run_name)`` unless ``run_name is ...``."""
    cfg: dict[str, Any] = {}
    if handler:
        cfg["callbacks"] = [LangGraphCallbackHandler()]

    async def go():
        return await graph.ainvoke({"x": "q", "out": []}, config=cfg)

    tracing._last_trace = None
    if run_name is ...:
        asyncio.run(go())
    else:
        with tracing.run(run_name):
            asyncio.run(go())
    trace = tracing._get_last_trace()
    assert trace is not None
    return trace


def _spans(trace):
    return trace["spans"]


def _by_id(trace):
    return {s["span_id"]: s for s in _spans(trace)}


def _agents(trace):
    return [s for s in _spans(trace) if s["type"] == "agent"]


def _ancestors(trace, span):
    by_id = _by_id(trace)
    out = []
    cur = by_id.get(span.get("parent_span_id"))
    while cur is not None:
        out.append(cur)
        cur = by_id.get(cur.get("parent_span_id"))
    return out


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for k in ("OTEL_SERVICE_NAME", "OTEL_RESOURCE_ATTRIBUTES", "MORSE_AGENT_IDENTITY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(sys, "argv", ["pytest"])
    monkeypatch.setattr(sys.modules["__main__"], "__spec__", None, raising=False)


# --- AID-01 identity ladder --------------------------------------------------


@pytest.mark.req("AID-01")
def test_compiled_graph_without_run_is_named_by_the_graph_not_the_script(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["drv_lg.py"])
    trace = _invoke(_react_graph("research-assistant"))
    assert trace["trace"]["agent"] == "research-assistant"
    assert trace["trace"]["metadata"] == {"identity_source": "framework"}
    assert [s["name"] for s in _agents(trace)] == ["research-assistant"]


@pytest.mark.req("AID-01")
def test_graph_beats_service_name(monkeypatch):
    monkeypatch.setenv("OTEL_SERVICE_NAME", "svc")
    trace = _invoke(_react_graph("research-assistant"))
    assert trace["trace"]["agent"] == "research-assistant"


@pytest.mark.req("AID-01")
def test_service_name_beats_entry_script(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["drv_lg.py"])
    monkeypatch.setenv("OTEL_SERVICE_NAME", "svc")
    trace = _invoke(_react_graph("LangGraph"))  # generic default: not a real name
    assert trace["trace"]["agent"] == "svc"
    assert trace["trace"]["metadata"] == {"identity_source": "service_name"}


@pytest.mark.req("AID-01")
def test_nothing_discoverable_falls_back_to_entry_script_and_is_flagged(monkeypatch):
    """oss1 collapsed ten agents into ``run_service`` (argv[0]); now it says so."""
    monkeypatch.setattr(sys, "argv", ["run_service.py"])
    trace = _invoke(_react_graph("LangGraph"))
    assert trace["trace"]["agent"] == "run_service"
    assert trace["trace"]["metadata"] == {"identity_source": "entry_script"}


@pytest.mark.req("AID-01")
def test_run_name_alone_is_explicit():
    with tracing.run("my-agent"):
        pass
    trace = tracing._get_last_trace()
    assert trace["trace"]["agent"] == "my-agent"
    assert trace["trace"]["metadata"] == {"identity_source": "run"}


# --- AID-02 / AID-05 hedge-fund fan-out -------------------------------------


@pytest.mark.req("AID-02", "AID-05")
def test_fourteen_analyst_fan_out_is_not_176_agents():
    """group D scenario 3: 218 spans, 176 agent-typed, linear spawn chain."""
    trace = _invoke(_fanout_graph("hedge-fund", ANALYSTS), run_name=None)
    names = [s["name"] for s in _agents(trace)]
    assert sorted(names) == sorted(["hedge-fund", *ANALYSTS]), names
    assert len(_agents(trace)) <= len(ANALYSTS) + 1  # the AID-02 bound
    # Plumbing is recorded, but as chain/node spans, never as agents.
    plumbing = [s for s in _spans(trace) if s["type"] == "custom"]
    assert plumbing
    assert {"manager"} <= {s["name"] for s in plumbing}
    assert not {"chain", "__start__", "tools", "RunnableSequence"} & set(names)


@pytest.mark.req("AID-05")
def test_parallel_branches_never_cross_parent_and_nothing_is_sentinel_parented():
    """group D finding 4/5: ``ben_graham`` under ``warren_buffett``'s chain."""
    trace = _invoke(_fanout_graph("hedge-fund", ANALYSTS[:6]), run_name=None)
    by_id = _by_id(trace)
    roots = [s for s in _spans(trace) if not s.get("parent_span_id")]
    assert len(roots) == 1 and roots[0]["name"] == "hedge-fund"
    for s in _spans(trace):
        assert not s.get("parent_span_id") or s["parent_span_id"] in by_id, s["name"]
    analyst_agents = {s["name"]: s for s in _agents(trace) if s["name"] in ANALYSTS}
    assert set(analyst_agents) == set(ANALYSTS[:6])
    for a in analyst_agents.values():
        assert a["parent_span_id"] == roots[0]["span_id"]  # siblings, no a->b chain
    llms = [s for s in _spans(trace) if s["type"] == "llm"]
    assert len(llms) == 6
    for llm in llms:
        owners = [
            x["name"] for x in _ancestors(trace, llm) if x["name"] in analyst_agents
        ]
        assert len(owners) == 1, owners  # exactly its own analyst, no other branch


@pytest.mark.req("AID-05")
def test_lcel_chain_nests_model_under_the_chain_under_the_analyst():
    trace = _invoke(_fanout_graph("g", ["a1"], async_nodes=False), run_name=None)
    by_id = _by_id(trace)
    (llm,) = [s for s in _spans(trace) if s["type"] == "llm"]
    chain = by_id[llm["parent_span_id"]]
    assert chain["type"] == "custom"
    assert chain["metadata"]["node_kind"] == "chain"
    prompt = next(s for s in _spans(trace) if s["name"] == "ChatPromptTemplate")
    assert prompt["parent_span_id"] == chain["span_id"]
    assert by_id[chain["parent_span_id"]]["name"] == "a1"


@pytest.mark.req("AID-05")
def test_llm_cost_rolls_up_to_the_analyst_that_made_the_call():
    trace = _invoke(_fanout_graph("g", ANALYSTS[:4]), run_name=None)
    agents = {s["name"]: s["span_id"] for s in _agents(trace)}
    tokens: dict[str, int] = {}
    for llm in (s for s in _spans(trace) if s["type"] == "llm"):
        (owner,) = [x["name"] for x in _ancestors(trace, llm) if x["name"] in ANALYSTS]
        tokens[owner] = tokens.get(owner, 0) + llm["total_tokens"]
    assert tokens == {a: 15 for a in ANALYSTS[:4]}
    assert set(agents) >= set(tokens)


@pytest.mark.req("AID-02")
def test_react_agent_and_tools_nodes_are_spans_not_agents():
    trace = _invoke(_react_graph("research-assistant"))
    assert [s["name"] for s in _agents(trace)] == ["research-assistant"]
    nodes = {
        s["name"]: s["metadata"]["node_kind"]
        for s in _spans(trace)
        if s["type"] == "custom" and s["metadata"]["node_kind"] == "node"
    }
    assert nodes == {"agent": "node", "tools": "node"}
    (llm,) = [s for s in _spans(trace) if s["type"] == "llm"]
    assert "agent" in {x["name"] for x in _ancestors(trace, llm)}


# --- MHQ-1402 / group B F1 regression, under the new shape ---------------------


@pytest.mark.req("AID-05")
def test_async_langgraph_without_run_records_children_under_the_root():
    trace = _invoke(_fanout_graph("async-app", ["a1", "a2"]))
    assert trace["trace"]["agent"] == "async-app"
    assert len(_spans(trace)) > 2  # the F1 failure was exactly two root spans
    assert len([s for s in _spans(trace) if s["type"] == "llm"]) == 2
    by_id = _by_id(trace)
    assert all(
        not s.get("parent_span_id") or s["parent_span_id"] in by_id
        for s in _spans(trace)
    )


# --- AID-04 run() merges ------------------------------------------------------


@pytest.mark.req("AID-04")
def test_run_name_becomes_app_label_and_discovered_root_keeps_identity():
    trace = _invoke(_fanout_graph("triage", ["a1"]), run_name="airline-support")
    assert trace["trace"]["agent"] == "triage"
    assert trace["trace"]["metadata"] == {
        "identity_source": "framework",
        "morse_app": "airline-support",
    }
    assert [s["name"] for s in _agents(trace)] == ["triage", "a1"]
    assert "airline-support" not in {s["name"] for s in _spans(trace)}


@pytest.mark.req("AID-04")
def test_generic_graph_name_keeps_the_run_name_and_sets_no_app():
    trace = _invoke(_react_graph("LangGraph"), run_name="airline-support")
    assert trace["trace"]["agent"] == "airline-support"
    assert trace["trace"]["metadata"] == {"identity_source": "run"}


@pytest.mark.req("AID-04")
def test_explicit_handler_name_is_not_overridden_by_graph_name():
    handler = LangGraphCallbackHandler("explicit")

    async def go():
        await _react_graph("graph-name").ainvoke(
            {"x": "q", "out": []}, config={"callbacks": [handler]}
        )

    tracing._last_trace = None
    asyncio.run(go())
    assert tracing._get_last_trace()["trace"]["agent"] == "explicit"


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
    yield eng
    eng.reset_for_testing()


@pytest.mark.req("AID-01", "AID-04")
def test_otlp_root_span_carries_morse_app_and_identity_source(engine, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_service.py"])
    with tracing.run(None):
        pass
    entry = tracing._get_last_trace()
    assert entry["trace"]["metadata"] == {"identity_source": "entry_script"}
    # An explicit run name wrapped around a graph: the exporter sees both.
    handler = LangGraphCallbackHandler()
    with tracing.run("airline-support") as r:
        asyncio.run(
            _react_graph("triage").ainvoke(
                {"x": "q", "out": []}, config={"callbacks": [handler]}
            )
        )
        trace_id = r.trace_id
    engine.provider.force_flush()
    spans = [
        s for s in engine.recorder.spans if s.context.trace_id == int(trace_id, 16)
    ]
    root = next(s for s in spans if s.parent is None or s.name == "triage")
    attrs = dict(root.attributes)
    assert attrs["morse.agent"] == "triage"
    assert attrs["morse.app"] == "airline-support"
    assert attrs["morse.identity_source"] == "framework"


# --- AID-09 legacy opt-out ------------------------------------------------------


@pytest.mark.req("AID-09")
def test_legacy_flag_restores_pre_v2_naming_and_shape(monkeypatch):
    monkeypatch.setenv("MORSE_AGENT_IDENTITY", "legacy")
    monkeypatch.setattr(sys, "argv", ["drv_lg.py"])
    trace = _invoke(_fanout_graph("hedge-fund", ANALYSTS[:3]), run_name=None)
    assert trace["trace"]["agent"] == "drv_lg"  # run(None) -> entry script, old way
    assert "metadata" not in trace["trace"]
    names = [s["name"] for s in _agents(trace)]
    assert len(names) > 3 + 1  # every chain/runnable is an agent span again
    assert not any(s["type"] == "custom" for s in _spans(trace))


@pytest.mark.req("AID-09")
def test_legacy_flag_keeps_run_name_as_the_agent(monkeypatch):
    monkeypatch.setenv("MORSE_AGENT_IDENTITY", "legacy")
    trace = _invoke(_react_graph("triage"), run_name="airline-support")
    assert trace["trace"]["agent"] == "airline-support"
    assert "metadata" not in trace["trace"]


# --- AID-04 / AID-05 OpenAI Agents hooks -----------------------------------------


def _start_trace(name):
    """A trace state with its root agent span on the stack (adapter test shape)."""
    run = tracing.run(name)
    run.__enter__()
    return run, tracing._active_trace.get(), run._root_span


@pytest.mark.req("AID-05")
def test_openai_agents_handoff_target_is_a_sub_agent_of_the_agent_that_handed_off():
    pytest.importorskip("agents")
    from amund.adapters import openai_agents as adapter

    r, state, root = _start_trace("triage")
    try:
        hooks = adapter._build_av_run_hooks()(state, root, {"name": "triage"}, None)
        triage, booking, refunds = (
            SimpleNamespace(name=n, model="gpt-4o")
            for n in ("triage", "booking", "refunds")
        )
        asyncio.run(hooks.on_handoff(SimpleNamespace(), triage, booking))
        asyncio.run(hooks.on_handoff(SimpleNamespace(), booking, refunds))
        agents = {s.name: s for s in state.completed_spans if s.type.value == "agent"}
        assert agents["booking"].parent_span_id == root.span_id
        assert agents["refunds"].parent_span_id == agents["booking"].span_id
    finally:
        r.__exit__(None, None, None)


@pytest.mark.req("AID-04")
def test_openai_agents_run_around_the_app_keeps_the_starting_agent_as_root():
    pytest.importorskip("agents")
    from amund.adapters import openai_agents as adapter

    with tracing.run("airline-support"):
        state = tracing.get_active_trace()
        adapter.adopt_native_root(state, "Triage Agent")
        assert state.agent_name == "Triage Agent"
        assert state.app == "airline-support"
        assert state.span_stack[0].name == "Triage Agent"
        # A nested run (guardrail, agent-as-tool) must not re-adopt.
        adapter.adopt_native_root(state, "Guardrail Agent")
        assert state.agent_name == "Triage Agent"
    trace = tracing._get_last_trace()
    assert trace["trace"]["metadata"]["morse_app"] == "airline-support"
