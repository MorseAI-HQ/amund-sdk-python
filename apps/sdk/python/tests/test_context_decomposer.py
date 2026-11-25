"""Tests for the shared OpenAI-format context segment decomposer.

``decompose_openai_messages`` is a pure function — these tests exercise it
directly without any I/O or AV SDK state.
"""

from __future__ import annotations

import hashlib

import pytest

from amund.context.decomposer import compute_composition, decompose_openai_messages

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _by_source(segs, source):
    return [s for s in segs if s["source"] == source]


# ---------------------------------------------------------------------------
# Basic happy-path tests
# ---------------------------------------------------------------------------


@pytest.mark.req("CTX-10")
def test_basic_user_message_emits_current_input():
    segs = decompose_openai_messages([{"role": "user", "content": "hello"}])
    inputs = _by_source(segs, "current_input")
    assert len(inputs) == 1
    assert inputs[0]["content"] == "hello"
    assert inputs[0]["tokens"] is None
    assert inputs[0]["content_hash"] == _sha("hello")


@pytest.mark.req("CTX-10")
def test_system_message_emits_system_prompt():
    segs = decompose_openai_messages(
        [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "hi"},
        ]
    )
    sys_segs = _by_source(segs, "system_prompt")
    assert len(sys_segs) == 1
    assert sys_segs[0]["content"] == "You are helpful"


@pytest.mark.req("CTX-10")
def test_tool_definitions_emitted_first():
    tools = [
        {
            "type": "function",
            "function": {"name": "get_weather", "description": "get weather"},
        }
    ]
    segs = decompose_openai_messages(
        [{"role": "user", "content": "what is the weather?"}],
        tools=tools,
    )
    assert segs[0]["source"] == "tool_descriptions"
    assert segs[0]["attrs"]["tool_count"] == 1
    assert "get_weather" in segs[0]["attrs"]["tool_names"]


@pytest.mark.req("CTX-10")
def test_tool_definitions_include_tool_names():
    tools = [
        {"type": "function", "function": {"name": "search"}},
        {"type": "function", "function": {"name": "summarize"}},
    ]
    segs = decompose_openai_messages(
        [{"role": "user", "content": "go"}],
        tools=tools,
    )
    td = _by_source(segs, "tool_descriptions")
    assert len(td) == 1
    assert set(td[0]["attrs"]["tool_names"]) == {"search", "summarize"}


@pytest.mark.req("CTX-10")
def test_prior_turns_are_chat_history():
    segs = decompose_openai_messages(
        [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "okay"},
            {"role": "user", "content": "second"},
        ]
    )
    history = _by_source(segs, "chat_history")
    current = _by_source(segs, "current_input")
    assert len(history) == 2
    assert len(current) == 1
    assert current[0]["content"] == "second"


@pytest.mark.req("CTX-10")
def test_tool_role_emits_tool_observation():
    segs = decompose_openai_messages(
        [
            {"role": "user", "content": "search something"},
            {"role": "assistant", "content": None},
            {"role": "tool", "content": "result data", "tool_call_id": "call_abc"},
            {"role": "user", "content": "now summarize"},
        ]
    )
    obs = _by_source(segs, "tool_observation")
    assert len(obs) == 1
    assert "result data" in obs[0]["content"]
    assert "call_abc" in obs[0]["content"]


@pytest.mark.req("CTX-10")
def test_position_ordering_is_sequential():
    tools = [{"type": "function", "function": {"name": "f"}}]
    segs = decompose_openai_messages(
        [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "prior"},
            {"role": "assistant", "content": "resp"},
            {"role": "user", "content": "now"},
        ],
        tools=tools,
    )
    positions = [s["position"] for s in segs]
    assert positions == list(range(len(segs)))


@pytest.mark.req("CTX-10")
def test_emission_order_tools_system_history_input():
    """tool_descriptions → system_prompt → chat_history → current_input."""
    tools = [{"type": "function", "function": {"name": "f"}}]
    segs = decompose_openai_messages(
        [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "resp"},
            {"role": "user", "content": "current"},
        ],
        tools=tools,
    )
    sources = [s["source"] for s in segs]
    assert sources[0] == "tool_descriptions"
    assert sources[1] == "system_prompt"
    # remaining: chat_history × 2, current_input
    assert sources[-1] == "current_input"
    assert all(s == "chat_history" for s in sources[2:-1])


@pytest.mark.req("CTX-10")
def test_content_list_parts_joined():
    """OpenAI vision format: content is a list of {type, text} blocks."""
    segs = decompose_openai_messages(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe this"},
                    {"type": "image_url", "image_url": {"url": "..."}},
                ],
            }
        ]
    )
    current = _by_source(segs, "current_input")
    assert len(current) == 1
    assert "describe this" in current[0]["content"]
    assert "[image]" in current[0]["content"]


@pytest.mark.req("CTX-10")
def test_long_content_truncated():
    """Long content is truncated; content_hash covers the full original text."""
    big = "x" * 5000
    segs = decompose_openai_messages([{"role": "user", "content": big}])
    assert len(segs) == 1
    seg = segs[0]
    assert seg["content_truncated"] is True
    assert len(seg["content"].encode("utf-8")) <= 4096
    assert seg["content_hash"] == _sha(big)


@pytest.mark.req("CTX-10")
def test_empty_messages_returns_empty():
    assert decompose_openai_messages([]) == []
    assert decompose_openai_messages(None) == []  # type: ignore[arg-type]


@pytest.mark.req("CTX-10")
def test_no_tools_no_tool_descriptions_segment():
    segs = decompose_openai_messages([{"role": "user", "content": "hi"}], tools=None)
    assert not _by_source(segs, "tool_descriptions")


@pytest.mark.req("CTX-10")
def test_malformed_input_returns_empty_not_raise():
    """Garbage input → empty list, no exception."""
    assert decompose_openai_messages("not a list") == []  # type: ignore[arg-type]
    assert decompose_openai_messages([{"role": None}]) == []


# ---------------------------------------------------------------------------
# LangChain BaseMessage duck-typing (no langchain import required)
# ---------------------------------------------------------------------------


@pytest.mark.req("CTX-10")
def test_langchain_message_objects_decomposed():
    """Messages with .type attribute (LangChain duck type) decomposed correctly."""

    class FakeHuman:
        type = "human"
        content = "what is 2+2"

    class FakeSystem:
        type = "system"
        content = "You are a math teacher"

    segs = decompose_openai_messages([FakeSystem(), FakeHuman()])
    sys_segs = _by_source(segs, "system_prompt")
    input_segs = _by_source(segs, "current_input")
    assert len(sys_segs) == 1
    assert len(input_segs) == 1
    assert sys_segs[0]["content"] == "You are a math teacher"
    assert input_segs[0]["content"] == "what is 2+2"


@pytest.mark.req("CTX-10")
def test_langchain_tool_message_emits_tool_observation():
    class FakeToolMsg:
        type = "tool"
        content = "search result"
        tool_call_id = "call_xyz"

    class FakeHuman:
        type = "human"
        content = "summarize"

    segs = decompose_openai_messages([FakeToolMsg(), FakeHuman()])
    obs = _by_source(segs, "tool_observation")
    assert len(obs) == 1
    assert "search result" in obs[0]["content"]


# ---------------------------------------------------------------------------
# compute_composition tests
# ---------------------------------------------------------------------------


def test_compute_composition_empty_segments_returns_zeroes():
    result = compute_composition([], 0)
    assert result == {
        "system": 0,
        "history": 0,
        "memory": 0,
        "retrieval": 0,
        "tools": 0,
        "current_input": 0,
    }


def test_compute_composition_single_system_segment():
    segments = [
        {
            "position": 0,
            "source": "system_prompt",
            "content": "x" * 100,
            "tokens": None,
        },
    ]
    result = compute_composition(segments, total_input_tokens=200)
    assert result["system"] == 200
    assert result["history"] == 0
    assert result["memory"] == 0


def test_compute_composition_proportional_allocation():
    segments = [
        {
            "position": 0,
            "source": "system_prompt",
            "content": "a" * 100,
            "tokens": None,
        },
        {"position": 1, "source": "chat_history", "content": "b" * 100, "tokens": None},
    ]
    result = compute_composition(segments, total_input_tokens=400)
    assert result["system"] == 200
    assert result["history"] == 200
    assert result["memory"] == 0


def test_compute_composition_uses_token_count_when_available():
    segments = [
        {"position": 0, "source": "system_prompt", "content": "a" * 50, "tokens": 60},
        {
            "position": 1,
            "source": "injected_memory",
            "content": "b" * 50,
            "tokens": 140,
        },
    ]
    result = compute_composition(segments, total_input_tokens=200)
    assert result["system"] == 60
    assert result["memory"] == 140


def test_compute_composition_buckets_tool_descriptions():
    segments = [
        {
            "position": 0,
            "source": "tool_descriptions",
            "content": "t" * 80,
            "tokens": None,
        },
        {
            "position": 1,
            "source": "tool_observation",
            "content": "o" * 20,
            "tokens": None,
        },
    ]
    result = compute_composition(segments, total_input_tokens=100)
    assert result["tools"] == 100
    assert result["memory"] == 0
    assert result["retrieval"] == 0


def test_compute_composition_zero_total_tokens_returns_zeroes():
    segments = [
        {"position": 0, "source": "system_prompt", "content": "x" * 50, "tokens": None},
    ]
    result = compute_composition(segments, total_input_tokens=0)
    assert all(v == 0 for v in result.values())
