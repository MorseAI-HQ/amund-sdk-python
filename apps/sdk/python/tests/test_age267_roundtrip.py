"""AGE-267 spike gate — round-trip tests (SDK side).

Covers:
  A. compute_composition() pure-function round-trip
  B. _scope_for_op() mapping round-trip

Part C (OTLP adapter mapping) lives in
  apps/api/tests/traces/test_age267_roundtrip.py
because the adapter is a Django app module.
"""

from amund.adapters.langgraph import _scope_for_op
from amund.context.decomposer import compute_composition

# ---------------------------------------------------------------------------
# A. Anthropic LLM span → context_composition
# ---------------------------------------------------------------------------


def test_compute_composition_roundtrip_explicit_tokens():
    """Segments with explicit token counts are bucketed directly."""
    segments = [
        {
            "source": "system_prompt",
            "content": "You are a helpful assistant.",
            "tokens": 10,
        },
        {"source": "chat_history", "content": "Previous messages...", "tokens": 50},
        {
            "source": "injected_memory",
            "content": "User fact: prefers Python.",
            "tokens": 20,
        },
    ]
    result = compute_composition(segments, total_input_tokens=80)

    assert result["system"] == 10
    assert result["history"] == 50
    assert result["memory"] == 20
    # Remaining buckets should be zero
    assert result["retrieval"] == 0
    assert result["tools"] == 0
    assert result["current_input"] == 0
    assert sum(result.values()) == 80


def test_compute_composition_all_buckets():
    """All six composition buckets receive correct token counts."""
    segments = [
        {"source": "system_prompt", "tokens": 5, "content": "sys"},
        {"source": "chat_history", "tokens": 30, "content": "hist"},
        {"source": "injected_memory", "tokens": 15, "content": "mem"},
        {"source": "retrieved_context", "tokens": 10, "content": "ret"},
        {"source": "tool_descriptions", "tokens": 8, "content": "tools"},
        {"source": "current_input", "tokens": 12, "content": "inp"},
    ]
    total = 5 + 30 + 15 + 10 + 8 + 12  # 80
    result = compute_composition(segments, total_input_tokens=total)

    assert result["system"] == 5
    assert result["history"] == 30
    assert result["memory"] == 15
    assert result["retrieval"] == 10
    assert result["tools"] == 8
    assert result["current_input"] == 12
    assert sum(result.values()) == total


def test_compute_composition_empty_returns_zeros():
    result = compute_composition([], total_input_tokens=100)
    assert all(v == 0 for v in result.values())


def test_compute_composition_zero_tokens_returns_zeros():
    segments = [{"source": "system_prompt", "content": "hello", "tokens": 10}]
    result = compute_composition(segments, total_input_tokens=0)
    assert all(v == 0 for v in result.values())


# ---------------------------------------------------------------------------
# B. LangGraph memory span → memory_scope
# ---------------------------------------------------------------------------


def test_scope_for_op_roundtrip():
    """Core op-type → scope mappings per ADR-023."""
    assert _scope_for_op("extract") == "working"
    assert _scope_for_op("consolidate") == "agent"
    assert _scope_for_op("retrieve", is_store_search=True) == "retrieval"
    assert _scope_for_op("prune") == "working"


def test_scope_for_op_retrieve_without_store_search():
    """'retrieve' without is_store_search still returns retrieval."""
    assert _scope_for_op("retrieve") == "retrieval"


def test_scope_for_op_store_search_flag_overrides():
    """is_store_search=True forces retrieval regardless of op name."""
    assert _scope_for_op("write", is_store_search=True) == "retrieval"
    assert _scope_for_op("unknown_op", is_store_search=True) == "retrieval"


def test_scope_for_op_unknown_defaults_to_working():
    assert _scope_for_op("unknown_op") == "working"
