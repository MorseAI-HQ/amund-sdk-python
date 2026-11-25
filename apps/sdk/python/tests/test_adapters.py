"""Tests for Anthropic and OpenAI adapters."""

from amund.adapters.anthropic import AnthropicAdapter
from amund.adapters.openai import OpenAIAdapter


class TestAnthropicAdapter:
    def setup_method(self):
        self.adapter = AnthropicAdapter()

    def test_extracts_basic_usage(self, anthropic_response):
        result = self.adapter.extract(anthropic_response)
        assert result["model"] == "claude-sonnet-4-6-20260217"
        assert result["input_tokens"] == 1200
        assert result["output_tokens"] == 350
        assert result["total_tokens"] == 1550
        assert "cost_usd" in result
        assert result["cost_usd"] > 0

    def test_extracts_cached_usage(self, anthropic_cached_response):
        result = self.adapter.extract(anthropic_cached_response)
        # input_tokens = 200 + 500 (cache_creation) + 300 (cache_read) = 1000
        assert result["input_tokens"] == 1000
        assert result["output_tokens"] == 350
        assert "cost_usd" in result

    def test_cost_calculation_accuracy(self, anthropic_response):
        result = self.adapter.extract(anthropic_response)
        # claude-sonnet-4-6: $3/MTok input, $15/MTok output
        expected_input_cost = (1200 / 1_000_000) * 3.0
        expected_output_cost = (350 / 1_000_000) * 15.0
        expected_total = round(expected_input_cost + expected_output_cost, 6)
        assert result["cost_usd"] == expected_total

    def test_handles_missing_usage(self):
        from types import SimpleNamespace

        response = SimpleNamespace(model="claude-sonnet-4-6", usage=None)
        result = self.adapter.extract(response)
        assert result == {"model": "claude-sonnet-4-6"}

    def test_handles_no_model(self):
        from types import SimpleNamespace

        response = SimpleNamespace(
            usage=SimpleNamespace(
                input_tokens=100,
                output_tokens=50,
                cache_creation_input_tokens=None,
                cache_read_input_tokens=None,
            )
        )
        result = self.adapter.extract(response)
        assert result["input_tokens"] == 100
        assert "cost_usd" not in result  # no model = no cost


class TestOpenAIAdapter:
    def setup_method(self):
        self.adapter = OpenAIAdapter()

    def test_extracts_chat_completions(self, openai_chat_response):
        result = self.adapter.extract(openai_chat_response)
        assert result["model"] == "gpt-4.1-2025-04-14"
        assert result["input_tokens"] == 800
        assert result["output_tokens"] == 200
        assert result["total_tokens"] == 1000
        assert "cost_usd" in result

    def test_extracts_responses_api(self, openai_responses_api):
        result = self.adapter.extract(openai_responses_api)
        assert result["model"] == "gpt-4.1"
        assert result["input_tokens"] == 800
        assert result["output_tokens"] == 200
        assert "cost_usd" in result

    def test_cost_with_cached_tokens(self, openai_responses_api):
        result = self.adapter.extract(openai_responses_api)
        # gpt-4.1: $2/MTok input, $8/MTok output, $0.5/MTok cached
        # 100 cached tokens at $0.5/MTok, 700 standard at $2/MTok
        standard_cost = (700 / 1_000_000) * 2.0
        cached_cost = (100 / 1_000_000) * 0.5
        output_cost = (200 / 1_000_000) * 8.0
        expected = round(standard_cost + cached_cost + output_cost, 6)
        assert result["cost_usd"] == expected

    def test_handles_missing_usage(self):
        from types import SimpleNamespace

        response = SimpleNamespace(model="gpt-4.1", usage=None)
        result = self.adapter.extract(response)
        assert result == {"model": "gpt-4.1"}
