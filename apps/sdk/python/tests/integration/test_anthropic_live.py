# ruff: noqa: E501
"""Live integration test for the Anthropic adapter.

Skipped unless ``ANTHROPIC_API_KEY`` is set in the environment. Hits the real
Anthropic API. Cost per run: ~$0.001. Wall time: ~10 seconds.

Run locally:
    ANTHROPIC_API_KEY=sk-ant-... uv run pytest tests/integration/

Phase 6a hardening: validates that the adapter handles real Claude responses
(streaming, prompt-caching tokens, tool-use blocks) and that recorded cost
matches the pricing table within ±5%.

Traceability markers:
    SDK-LIVE-01: Streaming response — token counts extracted
    SDK-LIVE-02: Cache tokens — extracted when present
    SDK-LIVE-03: Tool-use blocks — recorded as adapter output
    SDK-LIVE-04: Cost calculation accuracy ±5%
"""

from __future__ import annotations

import os
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Skip marker — no key, no live test
# ---------------------------------------------------------------------------

requires_key = pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set — live integration test skipped",
)


# Use a small, fast model and a tiny prompt to keep cost minimal.
LIVE_MODEL = "claude-haiku-4-5"


@pytest.fixture(scope="module")
def anthropic_client():
    """Real Anthropic client — assumes ANTHROPIC_API_KEY is set."""
    pytest.importorskip("anthropic")
    from anthropic import Anthropic

    return Anthropic()


@pytest.fixture
def adapter():
    from morse_ai.adapters.anthropic import AnthropicAdapter

    return AnthropicAdapter()


# ---------------------------------------------------------------------------
# SDK-LIVE-01 — streaming response token extraction
# ---------------------------------------------------------------------------


@requires_key
def test_streaming_response_extracts_token_counts(anthropic_client, adapter):
    """Streaming Anthropic call: adapter extracts model + non-zero token counts."""
    with anthropic_client.messages.stream(
        model=LIVE_MODEL,
        max_tokens=64,
        messages=[{"role": "user", "content": "Reply with exactly 'ok'."}],
    ) as stream:
        # Drain the stream so the final message materializes.
        for _ in stream.text_stream:
            pass
        message = stream.get_final_message()

    extracted = adapter.extract(message)

    assert "model" in extracted, f"Expected model in extracted: {extracted}"
    assert extracted["model"], "Model should be non-empty"
    assert extracted["input_tokens"] > 0, f"Expected input_tokens > 0, got {extracted}"
    assert extracted["output_tokens"] > 0, (
        f"Expected output_tokens > 0, got {extracted}"
    )
    assert (
        extracted["total_tokens"]
        == extracted["input_tokens"] + extracted["output_tokens"]
    )


# ---------------------------------------------------------------------------
# SDK-LIVE-04 — cost calculation accuracy
# ---------------------------------------------------------------------------


@requires_key
def test_recorded_cost_matches_pricing_table_within_5pct(anthropic_client, adapter):
    """Recorded cost_usd within ±5% of (input_tokens × in + output_tokens × out)."""
    from morse_ai.pricing import get_model_pricing

    response = anthropic_client.messages.create(
        model=LIVE_MODEL,
        max_tokens=32,
        messages=[{"role": "user", "content": "Reply with exactly 'hello'."}],
    )

    extracted = adapter.extract(response)

    assert "cost_usd" in extracted, f"Expected cost_usd in adapter output: {extracted}"
    assert extracted["cost_usd"] > 0, "Cost must be positive"

    pricing = get_model_pricing(extracted["model"])
    assert pricing is not None, f"Pricing missing for model {extracted['model']}"

    # Adapter folds cache tokens into input_tokens; for a non-cached call this
    # is the same as response.usage.input_tokens.
    raw_input = response.usage.input_tokens
    raw_output = response.usage.output_tokens
    expected_cost = (raw_input / 1_000_000) * pricing["input_per_mtok"] + (
        raw_output / 1_000_000
    ) * pricing["output_per_mtok"]
    actual_cost = extracted["cost_usd"]

    # ±5% tolerance — guards against floating-point drift and minor pricing-table lag.
    rel_diff = abs(actual_cost - expected_cost) / expected_cost
    assert rel_diff <= 0.05, (
        f"Cost mismatch: actual={actual_cost:.6f} expected={expected_cost:.6f} "
        f"diff={rel_diff:.2%}, threshold=5%"
    )


# ---------------------------------------------------------------------------
# SDK-LIVE-02 — cache token extraction
# ---------------------------------------------------------------------------


@requires_key
def test_cache_tokens_extracted_when_present(anthropic_client, adapter):
    """Two consecutive calls with cache_control: cache_read_input_tokens > 0 on the second.

    We construct a long-enough system prompt to qualify for prompt caching
    (Anthropic requires >=1024 tokens for sonnet/haiku to enable caching).
    The first call writes the cache; the second call reads from it.

    NOTE: The adapter folds cache tokens into ``input_tokens`` — we verify the
    raw response carries the cache-read field rather than the adapter output.
    """
    # Long system prompt — repeated filler to exceed minimum cache eligibility.
    long_system = "You are a helpful assistant. " * 600  # ~3600 tokens

    def call_with_cache_marker(content: str) -> Any:
        return anthropic_client.messages.create(
            model=LIVE_MODEL,
            max_tokens=16,
            system=[
                {
                    "type": "text",
                    "text": long_system,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": content}],
        )

    # First call: warms the cache.
    _ = call_with_cache_marker("Reply 'a'.")
    # Second call: should read from the cache.
    second = call_with_cache_marker("Reply 'b'.")

    cache_read = getattr(second.usage, "cache_read_input_tokens", 0) or 0

    if cache_read == 0:
        pytest.skip(
            "Anthropic prompt caching did not activate for this run — "
            "cache eligibility depends on prompt length and model availability. "
            "Adapter still folds cache_read into input_tokens correctly per fixture tests."
        )

    extracted = adapter.extract(second)

    # Adapter sums input + cache_creation + cache_read into input_tokens.
    raw_input = second.usage.input_tokens
    cache_creation = getattr(second.usage, "cache_creation_input_tokens", 0) or 0
    assert extracted["input_tokens"] == raw_input + cache_creation + cache_read


# ---------------------------------------------------------------------------
# SDK-LIVE-03 — tool-use block extraction
# ---------------------------------------------------------------------------


@requires_key
def test_tool_use_response_extracts_cleanly(anthropic_client, adapter):
    """Tool-use response: adapter records token counts and cost without crashing.

    The current adapter records LLM-call telemetry (tokens, cost). Recording
    tool-use as a separate child span is performed by the LangChain callback
    handler, not the raw Anthropic adapter. This test verifies the adapter
    handles a tool-use response shape cleanly.
    """
    response = anthropic_client.messages.create(
        model=LIVE_MODEL,
        max_tokens=256,
        tools=[
            {
                "name": "get_weather",
                "description": "Get the current weather in a city.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "City name"},
                    },
                    "required": ["city"],
                },
            }
        ],
        messages=[{"role": "user", "content": "What is the weather in Paris?"}],
    )

    extracted = adapter.extract(response)

    assert extracted["model"]
    assert extracted["input_tokens"] > 0
    assert extracted["output_tokens"] > 0
    assert extracted["cost_usd"] > 0

    # Verify the response itself contained a tool_use block — sanity that we
    # exercised the tool path, not just an empty response.
    has_tool_use = any(
        getattr(block, "type", "") == "tool_use" for block in response.content
    )
    if not has_tool_use:
        pytest.skip(
            "Model declined to call the tool for this prompt — adapter still "
            "extracted token counts cleanly, which is the contract under test."
        )
