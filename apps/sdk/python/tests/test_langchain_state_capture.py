"""Tests for LangChain adapter state-capture coercion (AGE-260, Issue 3).

The bug: LangGraph passes node state as a TypedDict, a Pydantic v1/v2 model,
a dataclass, or sometimes an arbitrary object. The previous adapter dropped
anything that wasn't ``isinstance(x, dict)`` straight to ``None``, so many
LangGraph node spans showed ``input_data: null`` / ``output_data: null`` on
the dashboard.

These tests parameterize the supported coercions and verify the new
``_ensure_dict`` helper preserves real state across every shape we expect to
see in the wild — without ballooning the transport queue for pathological
multi-MB payloads.
"""

# ruff: noqa: E501
from __future__ import annotations

import dataclasses
import json
import uuid
from typing import Any, TypedDict

import pytest

from morse_ai import tracing
from morse_ai.adapters.langchain import (
    _MAX_STATE_PAYLOAD_BYTES,
    MorseCallbackHandler,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chain_serialized(name: str = "Node") -> dict:
    return {"id": ["langgraph", "nodes", name], "name": name}


def _capture_chain_span(
    handler: MorseCallbackHandler,
    inputs: Any,
    outputs: Any,
) -> dict:
    """Drive on_chain_start/end with the given inputs/outputs and return the span."""
    run_id = uuid.uuid4()
    with tracing.run("test-agent"):
        handler.on_chain_start(_chain_serialized("TestNode"), inputs, run_id=run_id)
        handler.on_chain_end(outputs, run_id=run_id)
    trace = tracing._get_last_trace()
    assert trace is not None
    agent_spans = [
        s
        for s in trace["spans"]
        if s.get("type") == "agent" and s.get("name") == "TestNode"
    ]
    assert len(agent_spans) == 1, (
        f"expected exactly one TestNode span, got {agent_spans}"
    )
    return agent_spans[0]


# ---------------------------------------------------------------------------
# 1. Plain dict — pass-through behaviour (regression guard).
# ---------------------------------------------------------------------------


def test_dict_input_and_output_captured_unchanged():
    handler = MorseCallbackHandler("test")
    span = _capture_chain_span(
        handler,
        inputs={"messages": [{"role": "user", "content": "hi"}], "step": 1},
        outputs={"messages": [{"role": "assistant", "content": "hello"}], "step": 2},
    )
    assert span["input_data"] == {
        "messages": [{"role": "user", "content": "hi"}],
        "step": 1,
    }
    assert span["output_data"] == {
        "messages": [{"role": "assistant", "content": "hello"}],
        "step": 2,
    }


# ---------------------------------------------------------------------------
# 2. TypedDict — runtime is dict, must pass through.
# ---------------------------------------------------------------------------


class AgentState(TypedDict):
    foo: int
    bar: str


def test_typed_dict_input_and_output_captured_unchanged():
    """TypedDict instances are ``dict`` at runtime — must round-trip as-is."""
    handler = MorseCallbackHandler("test")
    state_in: AgentState = {"foo": 1, "bar": "input"}  # type: ignore[assignment]
    state_out: AgentState = {"foo": 2, "bar": "output"}  # type: ignore[assignment]
    span = _capture_chain_span(handler, state_in, state_out)
    assert span["input_data"] == {"foo": 1, "bar": "input"}
    assert span["output_data"] == {"foo": 2, "bar": "output"}


# ---------------------------------------------------------------------------
# 3. Pydantic v2 model — coerced via .model_dump().
# ---------------------------------------------------------------------------


def test_pydantic_v2_model_captured_via_model_dump():
    """Verify the Pydantic v2 coercion path using a duck-typed model_dump().

    We don't depend on pydantic being installed (the SDK's runtime deps are
    minimal) — the coercion is structural: any object exposing a callable
    ``model_dump()`` that returns a dict is treated as a Pydantic v2 model.
    """

    class V2State:
        def __init__(self, question: str, turn_count: int) -> None:
            self.question = question
            self.turn_count = turn_count

        def model_dump(self) -> dict:
            return {"question": self.question, "turn_count": self.turn_count}

    handler = MorseCallbackHandler("test")
    span = _capture_chain_span(
        handler,
        inputs=V2State("what?", 3),
        outputs=V2State("what?", 4),
    )
    assert span["input_data"] == {"question": "what?", "turn_count": 3}
    assert span["output_data"] == {"question": "what?", "turn_count": 4}


def test_pydantic_v2_real_basemodel_if_installed():
    """If pydantic is installed, verify against a real BaseModel — otherwise skip."""
    pydantic = pytest.importorskip("pydantic")
    if not hasattr(pydantic, "BaseModel"):
        pytest.skip("pydantic.BaseModel not available")

    class State(pydantic.BaseModel):
        question: str
        turn_count: int = 0

    handler = MorseCallbackHandler("test")
    span = _capture_chain_span(
        handler,
        inputs=State(question="what?", turn_count=3),
        outputs=State(question="what?", turn_count=4),
    )
    assert span["input_data"] == {"question": "what?", "turn_count": 3}
    assert span["output_data"] == {"question": "what?", "turn_count": 4}


# ---------------------------------------------------------------------------
# 4. Pydantic v1-style model — coerced via .dict() (compat fallback).
# ---------------------------------------------------------------------------


def test_pydantic_v1_style_model_captured_via_dict():
    """Duck-typed Pydantic v1 model with .dict() + __fields__ is coerced."""

    class V1Model:
        __fields__ = {"name": object(), "value": object()}

        def __init__(self, name: str, value: int) -> None:
            self.name = name
            self.value = value

        def dict(self) -> dict:  # noqa: A003 — mirrors Pydantic v1
            return {"name": self.name, "value": self.value}

        # Notably: no model_dump — this is the v1 path.

    handler = MorseCallbackHandler("test")
    span = _capture_chain_span(
        handler,
        inputs=V1Model(name="in", value=10),
        outputs=V1Model(name="out", value=20),
    )
    assert span["input_data"] == {"name": "in", "value": 10}
    assert span["output_data"] == {"name": "out", "value": 20}


# ---------------------------------------------------------------------------
# 5. Dataclass — coerced via dataclasses.asdict().
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class DCState:
    step: int
    label: str
    nested: dict[str, Any] = dataclasses.field(default_factory=dict)


def test_dataclass_input_and_output_captured_via_asdict():
    handler = MorseCallbackHandler("test")
    span = _capture_chain_span(
        handler,
        inputs=DCState(step=1, label="in", nested={"k": "v"}),
        outputs=DCState(step=2, label="out", nested={"k": "w"}),
    )
    assert span["input_data"] == {
        "step": 1,
        "label": "in",
        "nested": {"k": "v"},
    }
    assert span["output_data"] == {
        "step": 2,
        "label": "out",
        "nested": {"k": "w"},
    }


# ---------------------------------------------------------------------------
# 6. Plain scalars / lists — wrapped in {"value": str(...)}, never None.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        "a plain string state",
        42,
        ["list", "of", "things"],
        ("tuples", "too"),
    ],
)
def test_non_dict_non_object_state_captured_as_value_wrapper(value: Any):
    """Anything we can't structurally coerce becomes {"value": str(...)} —
    NEVER None. The whole point of the AGE-260 fix is no silent drops."""
    handler = MorseCallbackHandler("test")
    span = _capture_chain_span(handler, inputs=value, outputs=value)
    assert span["input_data"] is not None
    assert span["output_data"] is not None
    assert span["input_data"] == {"value": str(value)}
    assert span["output_data"] == {"value": str(value)}


# ---------------------------------------------------------------------------
# 7. Genuine None — must remain None (not all callbacks carry payloads).
# ---------------------------------------------------------------------------


def test_none_input_remains_none():
    """A legitimately-None input/output stays None — we only fix the silent-drop
    case where there WAS state but the strict isinstance check rejected it."""
    handler = MorseCallbackHandler("test")
    run_id = uuid.uuid4()
    with tracing.run("test-agent"):
        handler.on_chain_start(_chain_serialized("EmptyNode"), None, run_id=run_id)
        # on_chain_end requires outputs param; explicitly pass None.
        handler.on_chain_end(None, run_id=run_id)
    trace = tracing._get_last_trace()
    assert trace is not None
    spans = [s for s in trace["spans"] if s.get("name") == "EmptyNode"]
    assert len(spans) == 1
    # When _ensure_dict returns None, the span never gets input_data /
    # output_data attributes set, so they're absent from the serialized payload
    # (or, defensively, explicitly None). Both shapes are acceptable.
    assert spans[0].get("input_data") is None
    assert spans[0].get("output_data") is None


# ---------------------------------------------------------------------------
# 8. Oversized payload — truncated with marker, NOT dropped to None.
# ---------------------------------------------------------------------------


def test_oversized_payload_is_truncated_not_dropped():
    """A multi-MB Pydantic dump must NOT ride through unchecked — but it also
    must NOT silently become None. We expect a truncation marker."""
    handler = MorseCallbackHandler("test")
    # Build a dict whose JSON encoding exceeds the cap.
    big_str = "x" * (_MAX_STATE_PAYLOAD_BYTES + 10_000)
    big = {"messages": [{"role": "user", "content": big_str}]}
    span = _capture_chain_span(handler, inputs=big, outputs=big)

    assert span["input_data"] is not None
    assert span["input_data"].get("_truncated") is True
    assert span["input_data"].get("_reason") == "payload_too_large"
    # The marker must include a preview so the dashboard still has something
    # human-readable to show.
    assert "preview" in span["input_data"]
    assert isinstance(span["input_data"]["preview"], str)
    assert len(span["input_data"]["preview"]) > 0
    # And metadata about the original size so users know what got cut.
    assert span["input_data"]["_original_size_bytes"] > _MAX_STATE_PAYLOAD_BYTES
    assert span["input_data"]["_max_bytes"] == _MAX_STATE_PAYLOAD_BYTES

    # Output is symmetric.
    assert span["output_data"].get("_truncated") is True


# ---------------------------------------------------------------------------
# 9. Mixed: dict containing a Pydantic model (e.g., AgentState with a
# pydantic-typed field). We don't recurse, but the top-level dict round-trips
# and the encoder fallback keeps the model field readable.
# ---------------------------------------------------------------------------


def test_dict_with_nested_object_round_trips_via_json_fallback():
    """A top-level dict containing arbitrary objects passes through —
    the outer dict shape is preserved, inner objects ride through the
    ``json.dumps(default=str)`` fallback used for the size check."""

    class Sub:
        def __init__(self, x: int) -> None:
            self.x = x

        def __repr__(self) -> str:
            return f"Sub(x={self.x})"

    handler = MorseCallbackHandler("test")
    span = _capture_chain_span(
        handler,
        inputs={"sub": Sub(1), "msg": "hi"},
        outputs={"sub": Sub(2), "msg": "ok"},
    )
    # Top-level dict shape preserved. What matters is that the span has
    # SOMETHING — not None. The inner Sub object stringifies through
    # default=str during size measurement and persists in the payload.
    assert span["input_data"] is not None
    assert span["input_data"].get("msg") == "hi"
    assert span["output_data"] is not None
    assert span["output_data"].get("msg") == "ok"


# ---------------------------------------------------------------------------
# 10. Direct unit tests on _ensure_dict (no callback plumbing).
# ---------------------------------------------------------------------------


def test_ensure_dict_returns_none_for_none():
    assert MorseCallbackHandler._ensure_dict(None) is None


def test_ensure_dict_passthrough_for_dict():
    obj = {"a": 1}
    assert MorseCallbackHandler._ensure_dict(obj) == {"a": 1}


def test_ensure_dict_unserializable_object_falls_back_to_str():
    class Weird:
        def __repr__(self) -> str:
            return "<weird>"

        # Make json.dumps fail by raising in __str__ of an attribute. Easiest:
        # raise from __repr__ via default=str path — instead, use a class that
        # str() works on so we hit the {"value": str(x)} branch.

    w = Weird()
    result = MorseCallbackHandler._ensure_dict(w)
    assert result is not None
    assert result == {"value": "<weird>"}


def test_ensure_dict_payload_size_marker_serializable():
    """The truncation marker must itself be JSON-encodable (no recursion)."""
    big_dict = {"data": "y" * (_MAX_STATE_PAYLOAD_BYTES + 100)}
    result = MorseCallbackHandler._ensure_dict(big_dict)
    assert result is not None
    # Should be encodable (no exceptions).
    encoded = json.dumps(result)
    assert "_truncated" in encoded
