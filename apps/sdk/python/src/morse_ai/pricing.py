"""LLM model pricing table and cost calculation."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("morse_ai")

# Load pricing data from bundled JSON
_PRICING_FILE = Path(__file__).parent / "data" / "pricing.json"
_pricing_cache: dict[str, Any] | None = None


def _load_pricing() -> dict[str, Any]:
    global _pricing_cache
    if _pricing_cache is None:
        with open(_PRICING_FILE) as f:
            _pricing_cache = json.load(f)
    return _pricing_cache


# Region/vendor prefixes stripped from routed model ids (LiteLLM/OpenRouter/Bedrock).
_REGION_PREFIXES = ("us.", "eu.", "apac.", "global.")
_VENDOR_PREFIXES = (
    "anthropic.",
    "meta.",
    "mistral.",
    "cohere.",
    "amazon.",
    "ai21.",
    "deepseek.",
    "openai.",
    "google.",
    "qwen.",
)


def _normalize_candidates(model: str) -> list[str]:
    """Canonical-key candidates for a (possibly provider-prefixed) model id.

    Mirrors the server cost_engine normalizer: handles "openai/gpt-4o",
    "openrouter/openai/gpt-4o", "vertex_ai/gemini-2.5-pro", and Bedrock
    "us.anthropic.claude-3-5-sonnet-20241022-v2:0". Most-specific first.
    """
    cands: list[str] = []
    seen: set[str] = set()

    def add(x: str) -> None:
        if x and x not in seen:
            seen.add(x)
            cands.append(x)

    add(model)
    if "/" in model:
        parts = model.split("/")
        for i in range(1, len(parts)):
            add("/".join(parts[i:]))

    base = cands[-1]
    for region in _REGION_PREFIXES:
        if base.startswith(region):
            base = base[len(region) :]
            add(base)
            break
    for vendor in _VENDOR_PREFIXES:
        if base.startswith(vendor):
            base = base[len(vendor) :]
            add(base)
            break
    if ":" in base:
        add(base.split(":", 1)[0])

    return cands


def _match_one(pricing: dict, model: str) -> dict[str, float] | None:
    # Exact match
    if model in pricing:
        return pricing[model]
    # Prefix match: "claude-opus-4-6-20260205" matches "claude-opus-4-6"
    best_key = None
    for key in pricing:
        if model.startswith(key) and (best_key is None or len(key) > len(best_key)):
            best_key = key
    if best_key is not None:
        return pricing[best_key]
    # Reverse: alias "claude-opus-4-6" matches "claude-opus-4-6-20260205"
    for key in pricing:
        if key.startswith(model):
            return pricing[key]
    return None


def get_model_pricing(model: str) -> dict[str, float] | None:
    """Look up pricing for a model ID.

    Handles full IDs (claude-opus-4-6-20260205), aliases (claude-opus-4-6), and
    provider-routed ids from LangGraph/LiteLLM/OpenRouter/Bedrock/Vertex
    (e.g. "openai/gpt-4o", "us.anthropic.claude-3-5-sonnet-20241022-v2:0") via
    exact, longest-prefix, and provider-prefix normalization.
    """
    pricing = _load_pricing()
    for cand in _normalize_candidates(model):
        hit = _match_one(pricing, cand)
        if hit is not None:
            return hit

    logger.debug("morse_ai: no pricing found for model=%s", model)
    return None


def calculate_cost(
    input_tokens: int,
    output_tokens: int,
    model: str,
    cached_input_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> float | None:
    """Calculate total cost in USD for a model invocation.

    Returns None if the model is not in the pricing table.
    """
    pricing = get_model_pricing(model)
    if pricing is None:
        return None

    input_per_mtok = pricing.get("input_per_mtok", 0)
    output_per_mtok = pricing.get("output_per_mtok", 0)
    cached_per_mtok = pricing.get("cached_per_mtok", input_per_mtok)
    cache_write_per_mtok = pricing.get("cache_write_per_mtok", 0)

    standard_input = max(0, input_tokens - cached_input_tokens)
    input_cost = (standard_input / 1_000_000) * input_per_mtok
    cached_cost = (cached_input_tokens / 1_000_000) * cached_per_mtok
    write_cost = (cache_write_tokens / 1_000_000) * cache_write_per_mtok
    output_cost = (output_tokens / 1_000_000) * output_per_mtok

    return input_cost + cached_cost + write_cost + output_cost
