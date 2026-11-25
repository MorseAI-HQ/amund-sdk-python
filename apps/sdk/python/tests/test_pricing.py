"""Tests for pricing table and cost calculation."""

import pytest

from amund.pricing import calculate_cost, get_model_pricing


class TestPricing:
    def test_exact_model_match(self):
        pricing = get_model_pricing("claude-sonnet-4-6")
        assert pricing is not None
        assert pricing["input_per_mtok"] == 3.0
        assert pricing["output_per_mtok"] == 15.0

    def test_prefix_match_with_date(self):
        pricing = get_model_pricing("claude-sonnet-4-6-20260217")
        assert pricing is not None
        assert pricing["input_per_mtok"] == 3.0

    def test_openai_model(self):
        pricing = get_model_pricing("gpt-4.1")
        assert pricing is not None
        assert pricing["input_per_mtok"] == 2.0
        assert pricing["output_per_mtok"] == 8.0

    def test_unknown_model_returns_none(self):
        pricing = get_model_pricing("unknown-model-xyz")
        assert pricing is None

    def test_calculate_cost_basic(self):
        cost = calculate_cost(
            input_tokens=1000,
            output_tokens=500,
            model="claude-sonnet-4-6",
        )
        assert cost is not None
        expected = (1000 / 1_000_000) * 3.0 + (500 / 1_000_000) * 15.0
        assert abs(cost - expected) < 1e-10

    def test_calculate_cost_with_cache(self):
        cost = calculate_cost(
            input_tokens=1000,
            output_tokens=500,
            model="claude-sonnet-4-6",
            cached_input_tokens=300,
            cache_write_tokens=200,
        )
        assert cost is not None
        # 700 standard input + 300 cached + 200 cache write + 500 output
        standard = (700 / 1_000_000) * 3.0
        cached = (300 / 1_000_000) * 0.3
        write = (200 / 1_000_000) * 3.75
        output = (500 / 1_000_000) * 15.0
        expected = standard + cached + write + output
        assert abs(cost - expected) < 1e-10

    def test_calculate_cost_unknown_model(self):
        cost = calculate_cost(
            input_tokens=1000,
            output_tokens=500,
            model="nonexistent-model",
        )
        assert cost is None


class TestProviderNormalization:
    def test_litellm_openai_prefix(self):
        p = get_model_pricing("openai/gpt-4o")
        assert p is not None and p["input_per_mtok"] == 2.5

    def test_openrouter_double_prefix(self):
        p = get_model_pricing("openrouter/openai/gpt-4o")
        assert p is not None and p["input_per_mtok"] == 2.5

    def test_bedrock_vendor_and_version_suffix(self):
        p = get_model_pricing("bedrock/anthropic.claude-sonnet-4-5-v1:0")
        assert p is not None and p["input_per_mtok"] == 3.0

    def test_bedrock_cross_region_prefix(self):
        p = get_model_pricing("us.anthropic.claude-3-haiku-20240307-v1:0")
        assert p is not None and p["input_per_mtok"] == 0.25

    def test_gpt5_corrected_price(self):
        p = get_model_pricing("gpt-5")
        assert p is not None
        assert p["input_per_mtok"] == 1.25
        assert p["output_per_mtok"] == 10.0

    def test_new_providers(self):
        assert get_model_pricing("gemini-2.5-pro")["input_per_mtok"] == 1.25
        assert get_model_pricing("vertex_ai/gemini-2.5-flash")["output_per_mtok"] == 2.5
        assert get_model_pricing("mistral-large-latest")["input_per_mtok"] == 2.0
        assert get_model_pricing("command-r-08-2024")["input_per_mtok"] == 0.15
        assert get_model_pricing("deepseek-chat")["output_per_mtok"] == 0.28


class TestTierSuffixNeverFallsBackToBaseModel:
    """MHQ-1264: gpt-5-nano was prefix-matched to gpt-5 and billed 25x over."""

    def test_gpt5_nano_and_mini_have_their_own_price(self):
        assert get_model_pricing("gpt-5-nano")["input_per_mtok"] == 0.05
        assert get_model_pricing("gpt-5-mini")["input_per_mtok"] == 0.25

    def test_dated_nano_matches_nano_not_base(self):
        assert get_model_pricing("gpt-5-nano-2025-08-07")["output_per_mtok"] == 0.4

    def test_unknown_tier_is_unpriced_not_base_priced(self):
        assert get_model_pricing("gpt-5-micro") is None


# WP-8 (2026-10-24): rows ported from the server catalog
# (apps/api/apps/traces/data/model_pricing.json, fix/wp8-cost-catalog 95400119c).
_WP8_ROWS = [
    # (model id as an SDK sees it, input, output, cached)
    ("gpt-6-astra", 10.0, 50.0, 1.0),
    ("gpt-5.1-2025-11-13", 1.25, 10.0, 0.125),
    ("gpt-5-pro", 15.0, 120.0, None),  # fell back to gpt-5 ($1.25/$10) before
    ("o3-pro", 20.0, 80.0, None),
    ("claude-mythos-5", 10.0, 50.0, 1.0),
    ("gemini-3.1-pro", 2.0, 12.0, 0.2),
    ("us.amazon.nova-pro-v1:0", 0.8, 3.2, 0.2),
    ("meta.llama4-scout-17b-instruct-v1:0", 0.17, 0.66, None),
    ("openai/gpt-oss-20b", 0.07, 0.3, None),
]


@pytest.mark.parametrize(("model", "inp", "out", "cached"), _WP8_ROWS)
def test_wp8_catalog_rows_resolve(model, inp, out, cached):
    pricing = get_model_pricing(model)
    assert pricing is not None, model
    assert pricing["input_per_mtok"] == inp
    assert pricing["output_per_mtok"] == out
    assert pricing.get("cached_per_mtok") == cached


def test_wp8_mythos_cache_write_is_priced():
    cost = calculate_cost(
        input_tokens=0,
        output_tokens=0,
        model="claude-mythos-5",
        cache_write_tokens=1_000_000,
    )
    assert cost == pytest.approx(12.5)
