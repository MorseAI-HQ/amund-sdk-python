"""Base adapter interface for LLM SDK response extraction."""

from __future__ import annotations

import logging
import math
from typing import Any

logger = logging.getLogger("morse_ai")


def _coerce_finite_positive(value: Any) -> float | None:
    """Coerce a candidate cost into a finite, non-negative float, else None.

    Guards against bools (``isinstance(True, int)`` is True in Python), NaN /
    +/-inf, negative numbers, and unparseable values. Returns ``None`` for any
    value that isn't a clean, usable USD cost.
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        cost = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(cost) or cost < 0:
        return None
    return cost


def extract_provider_reported_cost(response: Any) -> tuple[float, str] | None:
    """Pull a provider-reported *actual* per-generation cost from a response.

    Per cost-intelligence design D3/D11, some providers return the real billed
    USD cost for THIS generation — ground truth that beats any catalog
    computation. We currently understand two surfaces, both reached through the
    OpenAI-compatible / LangChain code paths (there is no dedicated adapter):

    - **OpenRouter** (when ``usage`` accounting is enabled): the cost appears at
      ``response.usage.cost`` (attr) or ``response["usage"]["cost"]`` (mapping).
      Via LangChain it surfaces under ``response.llm_output`` /
      ``generation_info`` extras (e.g. ``llm_output["usage"]["cost"]`` or a
      top-level ``llm_output["cost"]``).
    - **LiteLLM**: ``response._hidden_params["response_cost"]`` — a float USD.

    Returns ``(cost_usd, source)`` where ``source`` is ``"openrouter"`` or
    ``"litellm"``, or ``None`` when no usable cost is present. Never raises —
    instrumentation must not break the host. Missing keys, non-finite, or
    negative values all yield ``None``.
    """
    try:
        # 1) LiteLLM — response._hidden_params["response_cost"].
        hidden = getattr(response, "_hidden_params", None)
        if isinstance(hidden, dict):
            cost = _coerce_finite_positive(hidden.get("response_cost"))
            if cost is not None:
                return cost, "litellm"

        # 2) OpenRouter — usage.cost. Support both attribute-style usage
        #    objects (raw OpenAI/OpenRouter SDK response) and mapping-style.
        usage = getattr(response, "usage", None)
        if usage is None and isinstance(response, dict):
            usage = response.get("usage")
        if usage is not None:
            usage_cost: Any = None
            if isinstance(usage, dict):
                usage_cost = usage.get("cost")
            else:
                usage_cost = getattr(usage, "cost", None)
            cost = _coerce_finite_positive(usage_cost)
            if cost is not None:
                return cost, "openrouter"

        # 3) OpenRouter via LangChain — provider extras live on llm_output.
        #    Look for usage.cost nested under llm_output, then a top-level
        #    llm_output["cost"].
        llm_output = getattr(response, "llm_output", None)
        if isinstance(llm_output, dict):
            lo_usage = llm_output.get("usage")
            if isinstance(lo_usage, dict):
                cost = _coerce_finite_positive(lo_usage.get("cost"))
                if cost is not None:
                    return cost, "openrouter"
            cost = _coerce_finite_positive(llm_output.get("cost"))
            if cost is not None:
                return cost, "openrouter"
    except Exception as exc:  # noqa: BLE001 — never break the host
        logger.debug("morse_ai: extract_provider_reported_cost swallowed: %s", exc)
        return None

    return None


def merge_provider_reported_cost(
    metadata: dict[str, Any] | None, response: Any
) -> dict[str, Any] | None:
    """Return metadata with a provider-reported cost block merged in.

    Writes ``metadata["cost"]["provider_reported_usd"]`` and
    ``metadata["cost"]["source"]`` when a provider-reported cost is found,
    WITHOUT overwriting any other keys already in an existing ``cost`` block.
    Returns the (possibly newly-created) metadata dict, or the original
    ``metadata`` unchanged (possibly ``None``) when no cost is present.

    Never raises.
    """
    try:
        found = extract_provider_reported_cost(response)
        if found is None:
            return metadata
        cost_usd, source = found
        md: dict[str, Any] = metadata if isinstance(metadata, dict) else {}
        existing_cost = md.get("cost")
        cost_block: dict[str, Any] = (
            dict(existing_cost) if isinstance(existing_cost, dict) else {}
        )
        cost_block["provider_reported_usd"] = cost_usd
        cost_block["source"] = source
        md["cost"] = cost_block
        return md
    except Exception as exc:  # noqa: BLE001 — never break the host
        logger.debug("morse_ai: merge_provider_reported_cost swallowed: %s", exc)
        return metadata


class BaseAdapter:
    """Extract telemetry data from an LLM SDK response.

    Each adapter normalizes provider-specific response shapes into
    a uniform dict that can be passed to morse_ai.record().
    """

    def extract(self, response: Any) -> dict[str, Any]:
        """Extract telemetry fields from a provider SDK response.

        Returns a dict with any of:
            model: str
            input_tokens: int
            output_tokens: int
            total_tokens: int
            cost_usd: float
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement extract()")
