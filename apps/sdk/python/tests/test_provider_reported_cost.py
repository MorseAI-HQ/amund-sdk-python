"""Tests for provider-reported actual cost extraction (cost-intelligence D3/D11).

Covers the shared helper plus the OpenAI-compatible (OpenRouter) and LangChain
(OpenRouter/LiteLLM) on-LLM-end write paths. Canonical metadata contract:
    metadata["cost"]["provider_reported_usd"]  -> float USD for THIS generation
    metadata["cost"]["source"]                 -> "openrouter" | "litellm"
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from morse_ai.adapters.base import (
    extract_provider_reported_cost,
    merge_provider_reported_cost,
)
from morse_ai.adapters.openai import OpenAIAdapter

# ---------------------------------------------------------------------------
# Helper-level tests
# ---------------------------------------------------------------------------


class TestExtractProviderReportedCost:
    def test_openrouter_usage_cost_attr(self):
        # OpenRouter raw SDK shape: response.usage.cost (attribute).
        response = SimpleNamespace(
            model="openrouter/auto",
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, cost=0.0123),
        )
        assert extract_provider_reported_cost(response) == (0.0123, "openrouter")

    def test_openrouter_usage_cost_mapping(self):
        # response["usage"]["cost"] (mapping-style response).
        response = {"usage": {"cost": 0.5}}
        assert extract_provider_reported_cost(response) == (0.5, "openrouter")

    def test_litellm_response_cost(self):
        # LiteLLM: response._hidden_params["response_cost"].
        response = SimpleNamespace(
            model="gpt-4o-mini",
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
            _hidden_params={"response_cost": 0.000042},
        )
        assert extract_provider_reported_cost(response) == (0.000042, "litellm")

    def test_litellm_takes_precedence_over_usage_cost(self):
        # When both present, LiteLLM's explicit response_cost wins (checked first).
        response = SimpleNamespace(
            usage=SimpleNamespace(cost=0.9),
            _hidden_params={"response_cost": 0.1},
        )
        assert extract_provider_reported_cost(response) == (0.1, "litellm")

    def test_langchain_llm_output_nested_usage_cost(self):
        # OpenRouter via LangChain: llm_output["usage"]["cost"].
        response = SimpleNamespace(llm_output={"usage": {"cost": 0.07}})
        assert extract_provider_reported_cost(response) == (0.07, "openrouter")

    def test_langchain_llm_output_top_level_cost(self):
        response = SimpleNamespace(llm_output={"cost": 0.02})
        assert extract_provider_reported_cost(response) == (0.02, "openrouter")

    def test_missing_cost_returns_none(self):
        response = SimpleNamespace(
            model="gpt-4o",
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
        )
        assert extract_provider_reported_cost(response) is None

    def test_no_usage_no_hidden_params_returns_none(self):
        response = SimpleNamespace(model="gpt-4o")
        assert extract_provider_reported_cost(response) is None

    def test_negative_cost_returns_none(self):
        response = SimpleNamespace(usage=SimpleNamespace(cost=-1.0))
        assert extract_provider_reported_cost(response) is None

    def test_nan_cost_returns_none(self):
        response = SimpleNamespace(usage=SimpleNamespace(cost=float("nan")))
        assert extract_provider_reported_cost(response) is None

    def test_inf_cost_returns_none(self):
        response = SimpleNamespace(_hidden_params={"response_cost": float("inf")})
        assert extract_provider_reported_cost(response) is None

    def test_bool_cost_returns_none(self):
        # bool is an int subclass; must not be coerced to 1.0/0.0.
        response = SimpleNamespace(usage=SimpleNamespace(cost=True))
        assert extract_provider_reported_cost(response) is None

    def test_zero_cost_is_valid(self):
        # 0.0 is a legitimate (free) billed cost — not None.
        response = SimpleNamespace(_hidden_params={"response_cost": 0.0})
        assert extract_provider_reported_cost(response) == (0.0, "litellm")

    def test_unparseable_cost_returns_none(self):
        response = SimpleNamespace(usage=SimpleNamespace(cost="not-a-number"))
        assert extract_provider_reported_cost(response) is None

    def test_never_raises_on_garbage_response(self):
        # Object whose attribute access blows up must not propagate.
        class Boom:
            @property
            def _hidden_params(self) -> Any:  # noqa: D401
                raise RuntimeError("boom")

            @property
            def usage(self) -> Any:
                raise RuntimeError("boom")

        assert extract_provider_reported_cost(Boom()) is None


# ---------------------------------------------------------------------------
# merge_provider_reported_cost — non-destructive merge semantics
# ---------------------------------------------------------------------------


class TestMergeProviderReportedCost:
    def test_creates_metadata_and_cost_block(self):
        response = SimpleNamespace(_hidden_params={"response_cost": 0.001})
        md = merge_provider_reported_cost(None, response)
        assert md == {"cost": {"provider_reported_usd": 0.001, "source": "litellm"}}

    def test_preserves_existing_metadata_keys(self):
        response = SimpleNamespace(usage=SimpleNamespace(cost=0.01))
        md = merge_provider_reported_cost({"adapter": "openai"}, response)
        assert md["adapter"] == "openai"
        assert md["cost"] == {"provider_reported_usd": 0.01, "source": "openrouter"}

    def test_does_not_overwrite_other_cost_keys(self):
        response = SimpleNamespace(usage=SimpleNamespace(cost=0.01))
        existing = {"cost": {"catalog_estimate_usd": 0.02, "currency": "USD"}}
        md = merge_provider_reported_cost(existing, response)
        assert md["cost"]["catalog_estimate_usd"] == 0.02
        assert md["cost"]["currency"] == "USD"
        assert md["cost"]["provider_reported_usd"] == 0.01
        assert md["cost"]["source"] == "openrouter"

    def test_no_cost_leaves_metadata_unchanged(self):
        response = SimpleNamespace(model="gpt-4o")
        original = {"adapter": "openai"}
        assert merge_provider_reported_cost(original, response) is original

    def test_no_cost_with_none_metadata_returns_none(self):
        response = SimpleNamespace(model="gpt-4o")
        assert merge_provider_reported_cost(None, response) is None


# ---------------------------------------------------------------------------
# OpenAI adapter (OpenRouter via OpenAI-compatible client) write path
# ---------------------------------------------------------------------------


class TestOpenAIAdapterProviderCost:
    def setup_method(self):
        self.adapter = OpenAIAdapter()

    def test_openrouter_cost_populates_metadata(self):
        response = SimpleNamespace(
            model="openrouter/anthropic/claude-3.5-sonnet",
            usage=SimpleNamespace(
                prompt_tokens=100,
                completion_tokens=50,
                total_tokens=150,
                cost=0.00321,
            ),
        )
        result = self.adapter.extract(response)
        assert result["metadata"]["cost"]["provider_reported_usd"] == 0.00321
        assert result["metadata"]["cost"]["source"] == "openrouter"
        # Token extraction still works alongside.
        assert result["input_tokens"] == 100
        assert result["output_tokens"] == 50

    def test_litellm_cost_populates_metadata(self):
        response = SimpleNamespace(
            model="gpt-4o-mini",
            usage=SimpleNamespace(
                prompt_tokens=20, completion_tokens=10, total_tokens=30
            ),
            _hidden_params={"response_cost": 0.000099},
        )
        result = self.adapter.extract(response)
        assert result["metadata"]["cost"]["provider_reported_usd"] == 0.000099
        assert result["metadata"]["cost"]["source"] == "litellm"

    def test_no_provider_cost_writes_no_metadata(self):
        response = SimpleNamespace(
            model="gpt-4o",
            usage=SimpleNamespace(
                prompt_tokens=10, completion_tokens=5, total_tokens=15
            ),
        )
        result = self.adapter.extract(response)
        assert "metadata" not in result

    def test_non_finite_provider_cost_writes_no_metadata(self):
        response = SimpleNamespace(
            model="gpt-4o",
            usage=SimpleNamespace(
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
                cost=float("nan"),
            ),
        )
        result = self.adapter.extract(response)
        assert "metadata" not in result


# ---------------------------------------------------------------------------
# LangChain adapter on-LLM-end write path (OpenRouter / LiteLLM via LangChain)
# ---------------------------------------------------------------------------


def _make_llm_result(llm_output: dict[str, Any] | None, text: str = "hi") -> Any:
    """Build a minimal LLMResult-shaped object for the LangChain adapter."""
    gen = SimpleNamespace(text=text, message=None)
    return SimpleNamespace(generations=[[gen]], llm_output=llm_output)


class TestLangChainAdapterProviderCost:
    def _run_on_llm_end(self, response: Any):
        import morse_ai
        from morse_ai.adapters.langchain import MorseCallbackHandler
        from morse_ai.tracing import run

        captured: dict[str, Any] = {}
        handler = MorseCallbackHandler(agent_name="t")
        import uuid

        rid = uuid.uuid4()
        with run("t"):
            handler.on_llm_start(
                {"name": "llm"}, prompts=["p"], run_id=rid, invocation_params={}
            )
            # Grab the span before on_llm_end closes it.
            traced = handler._span_stack.get(str(rid))
            handler._on_llm_end_impl(response, rid)
            captured["metadata"] = traced._span.metadata if traced else None
        del morse_ai  # ensure import side-effects don't linger
        return captured["metadata"]

    def test_openrouter_via_langchain_populates_span_metadata(self):
        response = _make_llm_result({"usage": {"cost": 0.044}})
        md = self._run_on_llm_end(response)
        assert md is not None
        assert md["cost"]["provider_reported_usd"] == 0.044
        assert md["cost"]["source"] == "openrouter"

    def test_litellm_via_langchain_populates_span_metadata(self):
        # LiteLLM through LangChain still exposes _hidden_params on the result.
        response = _make_llm_result(None)
        response._hidden_params = {"response_cost": 0.00007}
        md = self._run_on_llm_end(response)
        assert md is not None
        assert md["cost"]["provider_reported_usd"] == 0.00007
        assert md["cost"]["source"] == "litellm"

    def test_no_provider_cost_leaves_span_metadata_none(self):
        response = _make_llm_result({"token_usage": {"prompt_tokens": 1}})
        md = self._run_on_llm_end(response)
        assert md is None or "cost" not in md
