"""Tests for pricing table and cost calculation."""

from morse_ai.pricing import calculate_cost, get_model_pricing


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
