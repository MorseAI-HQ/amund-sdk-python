"""OpenAI SDK adapter — extracts telemetry from OpenAI API responses.

Traces both the Chat Completions API (``chat.completions.create``) and the
Responses API (``responses.create``), which use different field names for the
same concepts. Streaming (``stream=True``) passes through without usage on
either API.

Three layers (mirrors the Anthropic adapter structure):

1. **OpenAIAdapter** — passive extractor. ``extract()`` pulls token counts,
   model, and cost from a response. ``extract_segments()`` decomposes the
   request kwargs into ``context_segments`` using the shared decomposer.

2. **wrap(client)** — RECOMMENDED. Returns a wrapper around the customer's
   ``openai.OpenAI()`` (or ``AsyncOpenAI``) client that instruments every
   ``chat.completions.create`` and ``responses.create`` call. Context segments
   captured automatically.

3. **install() / uninstall()** — module-level monkey-patch. Used by the
   auto-detect layer (``_internal.auto_detect``) for zero-config setup.
"""

from __future__ import annotations

import hashlib
import logging
import time
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from collections.abc import Iterator

from morse_ai._internal.active_framework import (
    clear_active_framework,
    set_active_framework,
)
from morse_ai._internal.errors import absorb_errors
from morse_ai.adapters.base import BaseAdapter, merge_provider_reported_cost
from morse_ai.context.decomposer import decompose_openai_messages
from morse_ai.pricing import calculate_cost
from morse_ai.spans import SpanData, SpanType
from morse_ai.tracing import (
    _finalize_span,
    _TraceState,
    get_active_trace,
)
from morse_ai.tracing import run as _av_run

logger = logging.getLogger("morse_ai")

# MHQ-774: cap on the serialized size of input_data/output_data captured on
# an LLM span — matches the Anthropic adapter's _CONTENT_BYTE_CAP so the
# server-side payload guard (AI-SEC-08, 256 KB on the span content columns)
# is approached at the same rate for every adapter.
_CONTENT_BYTE_CAP = 4096

_FRAMEWORK_NAME = "openai"
_INSTALL_FLAG = "_morse_ai_installed"
_WRAPPED_MARKER = "_morse_ai_wrapped"

_original_create = None
_original_acreate = None
_original_responses_create = None
_original_responses_acreate = None

_CHAT = "chat"
_RESPONSES = "responses"
_SPAN_NAMES = {
    _CHAT: "openai.chat.completions.create",
    _RESPONSES: "openai.responses.create",
}

ClientT = TypeVar("ClientT")

_summarize_mode: ContextVar[bool] = ContextVar(
    "morse_ai_openai_summarize_mode", default=False
)


@contextmanager
def summarize() -> Iterator[None]:
    """Mark wrapped OpenAI ``chat.completions.create`` calls as summarization.

    When this context is active, every call inside emits an additional sibling
    ``memory`` span with ``metadata.memory_op_type="consolidate"`` linked back
    to the LLM span. Mirrors the Anthropic adapter's ``summarize()`` pattern.
    Implements OI-MEM-01 (OpenAI memory span parity).

    Example::

        client = av_openai.wrap(openai.OpenAI())
        with av_openai.summarize():
            summary = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Summarize the above."}],
            )
    """
    token = _summarize_mode.set(True)
    try:
        yield
    finally:
        _summarize_mode.reset(token)


def _install_active() -> bool:
    try:
        import openai  # type: ignore[import-not-found]

        return bool(getattr(openai, _INSTALL_FLAG, False))
    except ImportError:
        return False


class OpenAIAdapter(BaseAdapter):
    """Extract token counts, model, and cost from OpenAI responses.

    Chat Completions (client.chat.completions.create):
        response.usage.prompt_tokens        -> int (input)
        response.usage.completion_tokens    -> int (output)
        response.usage.total_tokens         -> int
        response.usage.prompt_tokens_details.cached_tokens -> int

    Responses API (client.responses.create):
        response.usage.input_tokens         -> int
        response.usage.output_tokens        -> int
        response.usage.total_tokens         -> int
        response.usage.input_tokens_details.cached_tokens -> int

    The adapter auto-detects which API shape is present.
    """

    def extract(self, response: Any) -> dict[str, Any]:
        """Extract telemetry from an OpenAI response."""
        result: dict[str, Any] = {}

        # Model
        model = getattr(response, "model", None)
        if model:
            result["model"] = str(model)

        # Usage
        usage = getattr(response, "usage", None)
        if usage is None:
            return result

        # Detect API shape: Chat Completions uses prompt_tokens,
        # Responses API uses input_tokens
        input_tokens = (
            getattr(usage, "input_tokens", None)
            or getattr(usage, "prompt_tokens", None)
            or 0
        )
        output_tokens = (
            getattr(usage, "output_tokens", None)
            or getattr(usage, "completion_tokens", None)
            or 0
        )
        total_tokens = getattr(usage, "total_tokens", None) or (
            input_tokens + output_tokens
        )

        result["input_tokens"] = input_tokens
        result["output_tokens"] = output_tokens
        result["total_tokens"] = total_tokens

        # Extract cached tokens for accurate cost calculation
        cached_tokens = 0
        for details_attr in ("input_tokens_details", "prompt_tokens_details"):
            details = getattr(usage, details_attr, None)
            if details:
                cached_tokens = getattr(details, "cached_tokens", 0) or 0
                break

        # Reasoning tokens — o-series models expose completion_tokens_details
        # .reasoning_tokens. Also check output_tokens_details for Responses API.
        reasoning_tokens = 0
        for details_attr in ("completion_tokens_details", "output_tokens_details"):
            details = getattr(usage, details_attr, None)
            if details:
                reasoning_tokens = getattr(details, "reasoning_tokens", 0) or 0
                if reasoning_tokens:
                    break
        if reasoning_tokens:
            result["reasoning_tokens"] = reasoning_tokens

        # Cost calculation
        if model:
            cost = calculate_cost(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                model=str(model),
                cached_input_tokens=cached_tokens,
            )
            if cost is not None:
                result["cost_usd"] = round(cost, 6)

        # Cost-intelligence D3/D11: OpenRouter (and LiteLLM, when reached
        # through the OpenAI-compatible client) returns the real billed cost
        # for THIS generation. Surface it on the canonical metadata contract so
        # the server prefers it over any catalog computation. Never overwrites
        # an existing cost block's other keys; no-op when absent.
        merged = merge_provider_reported_cost(result.get("metadata"), response)
        if merged is not None:
            result["metadata"] = merged

        return result

    def extract_segments(
        self, kwargs: dict[str, Any], response: Any
    ) -> list[dict[str, Any]]:
        """Pre-segment the chat completions request kwargs into context_segments.

        Delegates to the shared OpenAI-format decomposer so LangChain,
        LangGraph, and bare OpenAI calls all produce the same segment shape.
        """
        if not isinstance(kwargs, dict):
            return []
        return decompose_openai_messages(
            kwargs.get("messages"),
            kwargs.get("tools"),
        )


def _extract_openai_summary_text(response: Any) -> str:
    """Return the first text content from an OpenAI Chat Completions response.

    Chat Completions: response.choices[0].message.content (str or None).
    Best-effort — returns "" on any unexpected shape.
    """
    try:
        choices = getattr(response, "choices", None)
        if not choices:
            return ""
        message = getattr(choices[0], "message", None)
        if message is None:
            return ""
        content = getattr(message, "content", None)
        return content if isinstance(content, str) else ""
    except Exception:  # noqa: BLE001
        return ""


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _truncate(text: str) -> tuple[str, bool]:
    """Truncate to _CONTENT_BYTE_CAP bytes, preserving a valid UTF-8 prefix.

    Returns (truncated_text, was_truncated). Mirrors the Anthropic adapter's
    helper of the same name (MHQ-772's pattern, generalized here for MHQ-774).
    """
    encoded = text.encode("utf-8", errors="replace")
    if len(encoded) <= _CONTENT_BYTE_CAP:
        return text, False
    truncated = encoded[:_CONTENT_BYTE_CAP].decode("utf-8", errors="ignore")
    return truncated, True


# ---------------------------------------------------------------------------
# MHQ-774 — input_data/output_data capture (mirrors the MHQ-772 fix already
# shipped for the raw Anthropic adapter). Before this, every LLM span on this
# adapter had input_data=None and output_data=None — context_segments (the
# pre-segmented breakdown) were captured, but the raw request/response
# content the trace-detail view's INPUT/OUTPUT panels read was never set.
# ---------------------------------------------------------------------------


def _safe_input_messages(kwargs: dict[str, Any] | None) -> dict[str, Any] | None:
    """Build a truncated, JSON-safe input_data dict from chat.completions.create
    kwargs.

    Captures ``messages`` as sent (OpenAI's Chat Completions format has no
    separate ``system`` param — a system prompt is just a message with
    ``role="system"`` in the same array, so no special-casing is needed here
    unlike the Anthropic adapter). Each message's plain-string ``content`` is
    truncated to ``_CONTENT_BYTE_CAP``; list-form content (multi-part /
    image blocks) is passed through as-is — those are typically small
    compared to a long text turn, and per-block truncation would require
    reconstructing OpenAI's content-part shape exactly, not worth the
    complexity for a display field. Returns None if there are no messages.
    """
    if not kwargs:
        return None
    messages = kwargs.get("messages")
    if not messages:
        return None

    safe_messages = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        content = msg.get("content")
        if isinstance(content, str):
            content, _ = _truncate(content)
        safe_messages.append({"role": role, "content": content})

    return {"messages": safe_messages} if safe_messages else None


def _extract_output_text(response: Any) -> str | None:
    """Extract and truncate the assistant's text output from a Chat
    Completions response.

    Handles both plain text content and tool_calls — a response that only
    calls a tool still gets a meaningful output_data value instead of
    showing nothing.
    """
    try:
        choices = getattr(response, "choices", None)
        if not choices:
            return None
        message = getattr(choices[0], "message", None)
        if message is None:
            return None
    except Exception:  # noqa: BLE001
        return None

    parts: list[str] = []
    content = getattr(message, "content", None)
    if isinstance(content, str) and content:
        parts.append(content)

    tool_calls = getattr(message, "tool_calls", None) or []
    for tool_call in tool_calls:
        function = getattr(tool_call, "function", None)
        name = getattr(function, "name", "tool") if function else "tool"
        arguments = getattr(function, "arguments", None) if function else None
        parts.append(
            f"[tool_call:{name}] {arguments}" if arguments else f"[tool_call:{name}]"
        )

    if not parts:
        return None
    joined = "\n".join(parts)
    truncated, _ = _truncate(joined)
    return truncated


def _record_summary_memory_span(
    *,
    state: _TraceState,
    llm_span: SpanData,
    response: Any,
    start_mono: float,
) -> None:
    """Emit sibling ``memory`` span when summarize-mode is active (OI-MEM-01)."""
    try:
        usage = getattr(response, "usage", None)
        summary_tokens = 0
        if usage is not None:
            summary_tokens = int(
                getattr(usage, "completion_tokens", None)
                or getattr(usage, "output_tokens", None)
                or 0
            )

        summary_text = _extract_openai_summary_text(response)
        summary_hash = _hash_text(summary_text)

        tokens_evicted = int(llm_span.input_tokens or 0)
        tokens_injected = summary_tokens

        memory_span = SpanData(
            trace_id=state.trace_id,
            type=SpanType.MEMORY,
            name="openai.summarize",
            parent_span_id=llm_span.span_id,
            metadata={
                "memory_op_type": "consolidate",
                "memory_scope": "agent",
                "memory_tokens_evicted": tokens_evicted,
                "memory_tokens_injected": tokens_injected,
                "summary_token_count": summary_tokens,
                "summary_content_hash": summary_hash,
            },
        )
        _finalize_span(memory_span, start_mono)
        state.completed_spans.append(memory_span)
    except Exception as exc:  # noqa: BLE001
        logger.debug(
            "morse_ai: openai summarize span recording swallowed error: %s", exc
        )


def _responses_messages(kwargs: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Normalize Responses API ``instructions`` + ``input`` into a
    Chat-Completions-style message list, so input capture and the shared
    context decomposer see one shape for both APIs.

    ``input`` is either a plain string (one user turn) or a list of input
    items; items carrying a ``role`` are kept as messages, other item types
    (function_call_output, reasoning, ...) are skipped for display.
    """
    if not kwargs:
        return []
    messages: list[dict[str, Any]] = []
    instructions = kwargs.get("instructions")
    if isinstance(instructions, str) and instructions:
        messages.append({"role": "system", "content": instructions})
    raw_input = kwargs.get("input")
    if isinstance(raw_input, str) and raw_input:
        messages.append({"role": "user", "content": raw_input})
    elif isinstance(raw_input, list):
        for item in raw_input:
            if isinstance(item, dict) and item.get("role"):
                messages.append(
                    {"role": item.get("role"), "content": item.get("content")}
                )
    return messages


def _extract_responses_output_text(response: Any) -> str | None:
    """Extract and truncate the text output of a Responses API response.

    Uses the SDK's ``output_text`` convenience property, plus any
    ``function_call`` output items so a tool-only turn isn't blank.
    """
    parts: list[str] = []
    try:
        text = getattr(response, "output_text", None)
        if isinstance(text, str) and text:
            parts.append(text)
        for item in getattr(response, "output", None) or []:
            if getattr(item, "type", None) != "function_call":
                continue
            name = getattr(item, "name", None) or "tool"
            arguments = getattr(item, "arguments", None)
            label = f"[tool_call:{name}]"
            parts.append(f"{label} {arguments}" if arguments else label)
    except Exception:  # noqa: BLE001
        return None
    if not parts:
        return None
    truncated, _ = _truncate("\n".join(parts))
    return truncated


def _record_llm_span_for_response(
    name: str,
    response: Any,
    start_mono: float,
    kwargs: dict[str, Any] | None = None,
    started_at_iso: str | None = None,
    api: str = _CHAT,
) -> None:
    state = get_active_trace()
    if state is None:
        return
    try:
        adapter = OpenAIAdapter()
        telemetry = adapter.extract(response)
        if api == _RESPONSES:
            # Normalize to the chat shape so input capture and segments are shared.
            kwargs = {
                "messages": _responses_messages(kwargs),
                "tools": (kwargs or {}).get("tools"),
            }
            output_text = _extract_responses_output_text(response)
        else:
            output_text = _extract_output_text(response)
        segments = adapter.extract_segments(kwargs or {}, response)

        span = SpanData(
            trace_id=state.trace_id,
            type=SpanType.LLM,
            name=name,
            parent_span_id=state.span_stack[-1].span_id if state.span_stack else None,
            model=telemetry.get("model"),
            input_tokens=telemetry.get("input_tokens"),
            output_tokens=telemetry.get("output_tokens"),
            total_tokens=telemetry.get("total_tokens"),
            cost_usd=telemetry.get("cost_usd"),
            cached_input_tokens=telemetry.get("cached_input_tokens"),
            reasoning_tokens=telemetry.get("reasoning_tokens"),
            context_segments=segments if segments else None,
            input_data=_safe_input_messages(kwargs),
            output_data={"content": output_text} if output_text else None,
        )
        if started_at_iso is not None:
            span.started_at = started_at_iso
        _finalize_span(span, start_mono)
        state.completed_spans.append(span)

        if _summarize_mode.get():
            _record_summary_memory_span(
                state=state,
                llm_span=span,
                response=response,
                start_mono=start_mono,
            )
    except Exception as exc:  # noqa: BLE001
        logger.debug("morse_ai: openai span recording swallowed error: %s", exc)


# ---------------------------------------------------------------------------
# wrap() — recommended public API (mirrors wrapAnthropic shape)
# ---------------------------------------------------------------------------


class _WrappedCompletions:
    def __init__(self, inner: Any, api: str = _CHAT) -> None:
        self._av_inner = inner
        self._av_api = api

    def create(self, *args: Any, **kwargs: Any) -> Any:
        span_name = _SPAN_NAMES[self._av_api]
        start_mono = time.monotonic()
        started_at_iso = datetime.now(UTC).isoformat()
        set_active_framework(_FRAMEWORK_NAME)
        try:
            if _install_active():
                return self._av_inner.create(*args, **kwargs)
            if get_active_trace() is not None:
                response = self._av_inner.create(*args, **kwargs)
                _record_llm_span_for_response(
                    span_name,
                    response,
                    start_mono,
                    kwargs,
                    started_at_iso,
                    self._av_api,
                )
                return response
            with _av_run(span_name):
                response = self._av_inner.create(*args, **kwargs)
                _record_llm_span_for_response(
                    span_name,
                    response,
                    start_mono,
                    kwargs,
                    started_at_iso,
                    self._av_api,
                )
                return response
        finally:
            clear_active_framework()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._av_inner, name)


class _WrappedAsyncCompletions:
    def __init__(self, inner: Any, api: str = _CHAT) -> None:
        self._av_inner = inner
        self._av_api = api

    async def create(self, *args: Any, **kwargs: Any) -> Any:
        span_name = _SPAN_NAMES[self._av_api]
        start_mono = time.monotonic()
        started_at_iso = datetime.now(UTC).isoformat()
        set_active_framework(_FRAMEWORK_NAME)
        try:
            if _install_active():
                return await self._av_inner.create(*args, **kwargs)
            if get_active_trace() is not None:
                response = await self._av_inner.create(*args, **kwargs)
                _record_llm_span_for_response(
                    span_name,
                    response,
                    start_mono,
                    kwargs,
                    started_at_iso,
                    self._av_api,
                )
                return response
            with _av_run(span_name):
                response = await self._av_inner.create(*args, **kwargs)
                _record_llm_span_for_response(
                    span_name,
                    response,
                    start_mono,
                    kwargs,
                    started_at_iso,
                    self._av_api,
                )
                return response
        finally:
            clear_active_framework()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._av_inner, name)


class _WrappedChat:
    def __init__(self, inner: Any, *, is_async: bool) -> None:
        self._av_inner = inner
        completions = getattr(inner, "completions", None)
        if completions is None:
            self._av_completions: Any = None
        elif is_async:
            self._av_completions = _WrappedAsyncCompletions(completions)
        else:
            self._av_completions = _WrappedCompletions(completions)

    @property
    def completions(self) -> Any:
        return self._av_completions

    def __getattr__(self, name: str) -> Any:
        return getattr(self._av_inner, name)


class _WrappedClient:
    def __init__(self, inner: Any) -> None:
        self._av_inner = inner
        is_async = type(inner).__name__.startswith("Async")
        chat = getattr(inner, "chat", None)
        self._av_chat: Any = _WrappedChat(chat, is_async=is_async) if chat else None
        responses = getattr(inner, "responses", None)
        if responses is None:
            self._av_responses: Any = None
        elif is_async:
            self._av_responses = _WrappedAsyncCompletions(responses, _RESPONSES)
        else:
            self._av_responses = _WrappedCompletions(responses, _RESPONSES)
        setattr(self, _WRAPPED_MARKER, True)

    @property
    def chat(self) -> Any:
        return self._av_chat

    @property
    def responses(self) -> Any:
        return self._av_responses

    def __getattr__(self, name: str) -> Any:
        return getattr(self._av_inner, name)


def wrap(client: ClientT) -> ClientT:
    """Wrap an OpenAI client so every ``chat.completions.create`` and
    ``responses.create`` call emits AV spans.

    Returns the wrapped client. Idempotent — wrapping an already-wrapped
    client is a no-op. Context segments are captured automatically from the
    messages and tools parameters on every call.

    Example::

        from morse_ai.adapters import openai as av_openai
        import openai

        client = av_openai.wrap(openai.OpenAI())
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "hello"}],
        )
    """
    if client is None:
        return client
    if getattr(client, _WRAPPED_MARKER, False):
        return client
    try:
        import openai  # type: ignore[import-not-found]

        if getattr(openai, _INSTALL_FLAG, False):
            return client
    except ImportError:
        pass
    try:
        return _WrappedClient(client)  # type: ignore[return-value]
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "morse_ai: failed to wrap OpenAI client (%s) — calls untraced", exc
        )
        return client


# ---------------------------------------------------------------------------
# install() / uninstall() — module-level monkey-patch (auto-detect compat)
# ---------------------------------------------------------------------------


def _wrap_create(original: Any, api: str = _CHAT) -> Any:
    span_name = _SPAN_NAMES[api]

    def traced_create(self: Any, *args: Any, **kwargs: Any) -> Any:
        start_mono = time.monotonic()
        started_at_iso = datetime.now(UTC).isoformat()
        set_active_framework(_FRAMEWORK_NAME)
        try:
            if get_active_trace() is not None:
                response = original(self, *args, **kwargs)
                _record_llm_span_for_response(
                    span_name,
                    response,
                    start_mono,
                    kwargs,
                    started_at_iso,
                    api,
                )
                return response
            with _av_run(span_name):
                response = original(self, *args, **kwargs)
                _record_llm_span_for_response(
                    span_name,
                    response,
                    start_mono,
                    kwargs,
                    started_at_iso,
                    api,
                )
                return response
        finally:
            clear_active_framework()

    traced_create.__wrapped__ = original
    return traced_create


def _wrap_acreate(original: Any, api: str = _CHAT) -> Any:
    span_name = _SPAN_NAMES[api]

    async def traced_acreate(self: Any, *args: Any, **kwargs: Any) -> Any:
        start_mono = time.monotonic()
        started_at_iso = datetime.now(UTC).isoformat()
        set_active_framework(_FRAMEWORK_NAME)
        try:
            if get_active_trace() is not None:
                response = await original(self, *args, **kwargs)
                _record_llm_span_for_response(
                    span_name,
                    response,
                    start_mono,
                    kwargs,
                    started_at_iso,
                    api,
                )
                return response
            with _av_run(span_name):
                response = await original(self, *args, **kwargs)
                _record_llm_span_for_response(
                    span_name,
                    response,
                    start_mono,
                    kwargs,
                    started_at_iso,
                    api,
                )
                return response
        finally:
            clear_active_framework()

    traced_acreate.__wrapped__ = original
    return traced_acreate


def is_installed() -> bool:
    try:
        import openai  # type: ignore[import-not-found]
    except ImportError:
        return False
    return getattr(openai, _INSTALL_FLAG, False)


@absorb_errors(fingerprint="morse_ai.adapters.openai.install")
def install() -> bool:
    """Monkey-patch the OpenAI SDK so every client is instrumented.

    Idempotent. Returns True if installed (newly or already), False if
    OpenAI isn't available.
    """
    global _original_create, _original_acreate
    global _original_responses_create, _original_responses_acreate

    try:
        import openai  # type: ignore[import-not-found]
    except ImportError:
        return False

    if getattr(openai, _INSTALL_FLAG, False):
        return True

    try:
        from openai.resources.chat.completions import (  # type: ignore[import-not-found]
            Completions,
        )

        _original_create = Completions.create
        Completions.create = _wrap_create(_original_create)  # type: ignore[method-assign]
    except (ImportError, AttributeError) as exc:
        logger.warning(
            "morse_ai: OpenAI adapter could not patch Completions.create: %s — "
            "OpenAI calls will run untraced",
            exc,
        )
        return False

    try:
        from openai.resources.chat.completions import (  # type: ignore[import-not-found]
            AsyncCompletions,
        )

        _original_acreate = AsyncCompletions.create
        AsyncCompletions.create = _wrap_acreate(_original_acreate)  # type: ignore[method-assign]
    except (ImportError, AttributeError):
        pass

    # Responses API (MHQ-1198). Optional: older openai SDKs predate it, so a
    # missing module leaves chat tracing installed rather than failing.
    try:
        from openai.resources.responses import (  # type: ignore[import-not-found]
            AsyncResponses,
            Responses,
        )

        _original_responses_create = Responses.create
        Responses.create = _wrap_create(_original_responses_create, _RESPONSES)  # type: ignore[method-assign]
        _original_responses_acreate = AsyncResponses.create
        AsyncResponses.create = _wrap_acreate(  # type: ignore[method-assign]
            _original_responses_acreate, _RESPONSES
        )
    except (ImportError, AttributeError) as exc:
        logger.debug("morse_ai: OpenAI Responses API not patched: %s", exc)

    setattr(openai, _INSTALL_FLAG, True)
    logger.info("morse_ai: OpenAI adapter installed")
    return True


@absorb_errors(fingerprint="morse_ai.adapters.openai.uninstall")
def uninstall() -> None:
    """Restore the original OpenAI methods. Idempotent."""
    global _original_create, _original_acreate
    global _original_responses_create, _original_responses_acreate

    try:
        import openai  # type: ignore[import-not-found]
    except ImportError:
        return

    if not getattr(openai, _INSTALL_FLAG, False):
        return

    if _original_create is not None:
        try:
            from openai.resources.chat.completions import (
                Completions,  # type: ignore[import-not-found]
            )

            Completions.create = _original_create  # type: ignore[method-assign]
        except (ImportError, AttributeError):
            pass
        _original_create = None

    if _original_acreate is not None:
        try:
            from openai.resources.chat.completions import (
                AsyncCompletions,  # type: ignore[import-not-found]
            )

            AsyncCompletions.create = _original_acreate  # type: ignore[method-assign]
        except (ImportError, AttributeError):
            pass
        _original_acreate = None

    if (
        _original_responses_create is not None
        or _original_responses_acreate is not None
    ):
        try:
            from openai.resources.responses import (  # type: ignore[import-not-found]
                AsyncResponses,
                Responses,
            )

            if _original_responses_create is not None:
                Responses.create = _original_responses_create  # type: ignore[method-assign]
            if _original_responses_acreate is not None:
                AsyncResponses.create = _original_responses_acreate  # type: ignore[method-assign]
        except (ImportError, AttributeError):
            pass
        _original_responses_create = None
        _original_responses_acreate = None

    setattr(openai, _INSTALL_FLAG, False)
