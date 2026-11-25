"""Anthropic SDK adapter — extracts telemetry from Claude API responses.

Three layers:

1. **AnthropicAdapter** — passive extractor. Given an `anthropic.types.Message`
   response, returns a dict of telemetry fields (model, tokens, cost). Used by
   the LangChain callback handler when tracing LangGraph/LangChain agents that
   call Claude.

2. **wrap(client, *, summarize=False)** — RECOMMENDED PUBLIC API. Returns a
   delegating wrapper around the customer's ``anthropic.Anthropic()`` client.
   Every ``messages.create`` call on the wrapped client emits an Amund
   ``llm`` span with pre-segmented ``context_segments`` (CTX-08) and, when
   ``summarize=True``, a sibling ``memory`` span (MEM-11). Mirrors the
   TypeScript SDK's ``wrapAnthropic()`` so the two SDKs have an identical
   integration shape. No global state, no monkey-patching of the Anthropic
   class — only the returned instance is instrumented.

3. **install() / uninstall()** — module-level monkey-patcher. Wraps
   ``anthropic.resources.messages.Messages.create`` (and the async variant)
   so every ``new Anthropic()`` client is automatically instrumented without
   the customer calling ``wrap()``. Used by the auto-detect layer
   (``amund._internal.auto_detect``) when the customer relies on
   ``AMUND_API_KEY``-only zero-config setup. Idempotent; safe to call
   repeatedly.

Why both ``wrap()`` and ``install()``?
- ``wrap()`` is the explicit, recommended path — same shape as the TS SDK,
  no global state, easy to opt out per-client (e.g., one wrapped client
  with ``summarize=True`` for a summarization helper, plus an unwrapped
  client for fire-and-forget calls).
- ``install()`` powers the zero-config auto-detect flow advertised in
  ``apps/sdk/python/src/amund/_internal/auto_detect.py``. Dropping it
  would silently break that path. The two coexist; ``install()`` is honest
  about being a compatibility shim and is not the path we document for new
  customers.

Both layers funnel into the same ``_record_llm_span_for_response`` helper,
so wire shape and behaviour stay identical regardless of which API the
customer chose. Wrapped under ``@absorb_errors`` so an installation
failure can never break the customer's Anthropic code — at worst, calls
go untraced.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, TypeVar

from amund._internal.active_framework import (
    clear_active_framework,
    get_llm_owner,
    set_active_framework,
)
from amund._internal.errors import absorb_errors
from amund.adapters.base import BaseAdapter
from amund.context.decomposer import compute_composition
from amund.pricing import calculate_cost
from amund.propagation import prepare_llm_call
from amund.spans import SpanData, SpanType
from amund.tracing import (
    _finalize_span,
    _TraceState,
    apply_end_user,
    get_active_trace,
)
from amund.tracing import (
    run as _av_run,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = logging.getLogger("amund")

# AGE-240: framework auto-discovery tag emitted via the thread-local set
# around every intercepted ``messages.create``. Read at trace emission time
# by ``TracedRun`` to stamp ``resource["morse_ai.framework"]``.
_FRAMEWORK_NAME = "anthropic"

# Sentinel attribute we set on the target module to mark it instrumented.
_INSTALL_FLAG = "_morse_ai_installed"

# Cached references to the originals so uninstall() can restore them.
_original_create = None
_original_acreate = None
_original_stream = None
_original_astream = None
_original_beta_create = None
_original_beta_acreate = None

# Per-segment content cap (bytes). Anything beyond is truncated and
# `content_truncated=True` is set so the server still has the prefix for
# debugging without blowing up payload size. Tokens are NOT computed
# client-side — the server tokenizes (CTX-04).
_CONTENT_BYTE_CAP = 4096

# Summarize-mode flag (MEM-11). When True, every Anthropic messages.create
# call inside the context emits an additional sibling `memory` span with
# `metadata.memory_op_type="consolidate"`, linking back to the LLM span via
# `parent_span_id`. The flag is opt-in only — set via the `summarize()`
# context manager OR by calling `wrap(client, summarize=True)`. Defaults
# False so existing callers see zero behavior change.
_summarize_mode: ContextVar[bool] = ContextVar(
    "morse_ai_anthropic_summarize_mode", default=False
)

# Marker attribute we set on the wrapper class instance so `wrap()` is
# idempotent — wrapping an already-wrapped client is a no-op.
_WRAPPED_MARKER = "_morse_ai_wrapped"

# Generic TypeVar so static type-checkers see ``wrap(Anthropic()) -> Anthropic``
# (we hand back a structural look-alike, not the literal Anthropic class).
ClientT = TypeVar("ClientT")


def _install_active() -> bool:
    """Return True iff ``install()`` has monkey-patched ``Messages.create``.

    Used by ``_WrappedMessages.create`` to dedup vs the global patch at
    call-time. wrap-time check isn't enough because customers commonly
    call ``wrap()`` before any SDK code triggers ``install()``.
    """
    try:
        import anthropic  # type: ignore[import-not-found]

        return bool(getattr(anthropic, _INSTALL_FLAG, False))
    except ImportError:
        return False


@contextmanager
def summarize() -> Iterator[None]:
    """Mark wrapped Anthropic ``messages.create`` calls as customer-side summarization.

    When this context is active, every Claude call made inside emits an
    additional sibling ``memory`` span with ``metadata.memory_op_type=
    "consolidate"``, ``parent_span_id`` linking back to the LLM span,
    ``metadata.summary_token_count`` from the response usage, and
    ``metadata.summary_content_hash`` (SHA-256 of the response's first text
    block) so the server can match it against subsequent LLM spans'
    ``context_segments`` and populate ``metadata.context_remediation
    .summary_segment_id``.

    Opt-in only. Two ways to enable summarize-mode:

      1. **Per-call override (this context manager):** preferred when a
         single client is used for both normal calls and a one-off summary::

            client = av_anthropic.wrap(Anthropic())
            with av_anthropic.summarize():
                summary = client.messages.create(...)

      2. **Per-client default:** set on the wrapped client itself, useful
         when you have a dedicated summarization helper::

            summarizer = av_anthropic.wrap(Anthropic(), summarize=True)
            summary = summarizer.messages.create(...)

    The ContextVar override (this manager) wins when both are set, matching
    the TypeScript SDK's behavior. Implements MEM-11 (Phase 9 Memory
    Profiler). Safe to nest; safe outside an active trace (no-op).
    Async-safe via ``ContextVar``.
    """
    token = _summarize_mode.set(True)
    try:
        yield
    finally:
        _summarize_mode.reset(token)


class AnthropicAdapter(BaseAdapter):
    """Extract token counts, model, and cost from Anthropic Message responses.

    Works with both sync and streaming responses. For streaming, pass
    the result of stream.get_final_message().

    Response shape (anthropic.types.Message):
        response.model              -> str (e.g., "claude-opus-4-6-20260205")
        response.usage.input_tokens -> int
        response.usage.output_tokens -> int
        response.usage.cache_creation_input_tokens -> Optional[int]
        response.usage.cache_read_input_tokens -> Optional[int]
    """

    def extract(self, response: Any) -> dict[str, Any]:
        """Extract telemetry from an Anthropic Message response."""
        result: dict[str, Any] = {}

        model = getattr(response, "model", None)
        if model:
            result["model"] = str(model)

        usage = getattr(response, "usage", None)
        if usage is None:
            return result

        input_tokens = getattr(usage, "input_tokens", 0) or 0
        output_tokens = getattr(usage, "output_tokens", 0) or 0
        cache_creation = getattr(usage, "cache_creation_input_tokens", 0) or 0
        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0

        result["input_tokens"] = input_tokens + cache_creation + cache_read
        result["output_tokens"] = output_tokens
        result["total_tokens"] = result["input_tokens"] + output_tokens
        # MHQ-1236: 0, not absent, when the response reports usage with no
        # cache activity -- "not caching" must differ from "not reported".
        result["cached_input_tokens"] = cache_read
        result["cache_write_tokens"] = cache_creation

        # Extended thinking: capture reasoning/thinking tokens when Anthropic
        # exposes a breakdown. API surface: usage.output_tokens_details
        # .thinking_tokens (or usage.thinking_tokens on some preview builds).
        thinking_tokens = 0
        output_details = getattr(usage, "output_tokens_details", None)
        if output_details is not None:
            thinking_tokens = getattr(output_details, "thinking_tokens", 0) or 0
        if not thinking_tokens:
            # Fallback for potential flat field on older preview builds.
            thinking_tokens = getattr(usage, "thinking_tokens", 0) or 0
        if thinking_tokens:
            result["reasoning_tokens"] = thinking_tokens

        if model:
            cost = calculate_cost(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                model=str(model),
                cached_input_tokens=cache_read,
                cache_write_tokens=cache_creation,
            )
            if cost is not None:
                result["cost_usd"] = round(cost, 6)

        return result

    def extract_segments(
        self, kwargs: dict[str, Any], response: Any
    ) -> list[dict[str, Any]]:
        """Pre-segment the request kwargs into a context_segments list.

        Anthropic-specific segmentation: walks `tools`, `system`, and `messages`
        to produce a canonically-ordered list (tool_descriptions →
        system_prompt → chat_history/current_input/tool_observation) tagged with
        cache_state where prompt-caching markers are visible.

        Returns [] (never raises) on malformed input — keeping the recorder
        non-invasive. Mirrors the passive `extract()` shape so the LangChain
        adapter can call this directly.
        """
        return _extract_anthropic_segments(kwargs, response)


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _truncate(text: str) -> tuple[str, bool]:
    """Truncate to _CONTENT_BYTE_CAP bytes, preserving a valid UTF-8 prefix.

    Returns (truncated_text, was_truncated).
    """
    encoded = text.encode("utf-8", errors="replace")
    if len(encoded) <= _CONTENT_BYTE_CAP:
        return text, False
    # Decode back, ignoring a possibly-split trailing multibyte sequence.
    truncated = encoded[:_CONTENT_BYTE_CAP].decode("utf-8", errors="ignore")
    return truncated, True


def _flatten_system(system: Any) -> tuple[str, bool]:
    """Return (concatenated_text, has_cache_control) for the `system` field.

    Anthropic accepts:
      - str: plain prompt
      - list[{type:"text", text, cache_control?}]: block form for prompt caching
    """
    if system is None:
        return "", False
    if isinstance(system, str):
        return system, False
    if isinstance(system, list):
        parts: list[str] = []
        has_cache = False
        for block in system:
            if not isinstance(block, dict):
                # Unrecognized block — skip silently rather than crash.
                continue
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)
            if block.get("cache_control"):
                has_cache = True
        return "".join(parts), has_cache
    # Unknown shape — coerce to string defensively.
    return str(system), False


def _tools_have_cache_control(tools: Any) -> bool:
    if not isinstance(tools, list):
        return False
    return any(isinstance(tool, dict) and tool.get("cache_control") for tool in tools)


def _stringify_message_content(content: Any) -> tuple[list[dict[str, str]], bool]:
    """Walk an Anthropic message's `content` and return per-block records.

    Anthropic message content can be:
      - str: plain text
      - list of blocks:
          {type: "text", text: ...}
          {type: "tool_use", id, name, input}
          {type: "tool_result", tool_use_id, content}
          {type: "image", ...}

    Returns a list of records each shaped
    {"kind": "<text|tool_result|tool_use|image|other>", "text": "..."},
    one per block, plus a bool flag for whether any tool_result blocks were seen
    (callers use this to split a single user message that contains tool_result
    blocks into a `tool_observation` segment).
    """
    records: list[dict[str, str]] = []
    has_tool_result = False
    if isinstance(content, str):
        records.append({"kind": "text", "text": content})
        return records, False
    if not isinstance(content, list):
        # Fall back to repr — keeps us moving on weird shapes.
        records.append({"kind": "other", "text": str(content)})
        return records, False
    for block in content:
        if isinstance(block, str):
            records.append({"kind": "text", "text": block})
            continue
        if not isinstance(block, dict):
            records.append({"kind": "other", "text": str(block)})
            continue
        btype = block.get("type")
        if btype == "text":
            records.append({"kind": "text", "text": str(block.get("text", ""))})
        elif btype == "tool_result":
            has_tool_result = True
            inner = block.get("content")
            if isinstance(inner, str):
                rendered = inner
            elif isinstance(inner, list):
                pieces: list[str] = []
                for ib in inner:
                    if isinstance(ib, dict) and ib.get("type") == "text":
                        pieces.append(str(ib.get("text", "")))
                    elif isinstance(ib, str):
                        pieces.append(ib)
                rendered = "\n".join(pieces)
            else:
                rendered = str(inner) if inner is not None else ""
            tool_use_id = block.get("tool_use_id", "")
            prefix = f"[tool_result tool_use_id={tool_use_id}]\n" if tool_use_id else ""
            records.append({"kind": "tool_result", "text": prefix + rendered})
        elif btype == "tool_use":
            name = block.get("name", "")
            inp = block.get("input")
            try:
                rendered_input = json.dumps(inp, sort_keys=True, default=str)
            except (TypeError, ValueError):
                rendered_input = str(inp)
            records.append(
                {
                    "kind": "tool_use",
                    "text": f"[tool_use name={name}]\n{rendered_input}",
                }
            )
        elif btype == "image":
            # We don't ship image bytes; record a stub so the segment count is honest.
            records.append({"kind": "image", "text": "[image]"})
        else:
            records.append({"kind": "other", "text": str(block)})
    return records, has_tool_result


def _make_segment(
    *,
    position: int,
    source: str,
    content: str,
    cache_state: str | None = None,
    attrs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    truncated_text, was_truncated = _truncate(content)
    segment: dict[str, Any] = {
        "position": position,
        "source": source,
        # Hash the FULL content (pre-truncation) so identical prompts dedupe even
        # when only the truncated prefix would otherwise differ.
        "content_hash": _hash_text(content),
        "content": truncated_text,
        "content_truncated": was_truncated,
        "tokens": None,  # server tokenizes (CTX-04)
        "tokens_estimated": False,
    }
    if cache_state is not None:
        segment["cache_state"] = cache_state
    if attrs:
        segment["attrs"] = attrs
    return segment


def _extract_anthropic_segments(
    kwargs: dict[str, Any] | None, response: Any
) -> list[dict[str, Any]]:
    """Build a context_segments list from Anthropic request kwargs + response.

    Cache-state heuristic:
      - A `system` block or `tools` entry with cache_control={"type":"ephemeral"}
        => emit cache_state="cache_write" on that segment. This is a request-side
        signal: the customer is *asking* for a cache write.
      - When response.usage.cache_read_input_tokens > 0 the cached prefix
        (system_prompt + tool_descriptions, in Anthropic's caching model)
        was actually served from cache. We tag those segments cache_state
        ="cache_hit" UNLESS they were already tagged cache_write above
        (the request-side signal wins for this turn).
      - Everything else stays cache_state=None (the server defaults to "cold").

    This is intentionally heuristic — Anthropic's API only reports cache hits at
    the prefix level, not per-block, so we attribute hits to the typically-cached
    prefix segments. Will improve once Anthropic exposes per-block hit data.
    """
    if not isinstance(kwargs, dict):
        return []

    try:
        segments: list[dict[str, Any]] = []
        position = 0

        tools = kwargs.get("tools")
        system = kwargs.get("system")
        messages = kwargs.get("messages")

        tools_has_cache_control = _tools_have_cache_control(tools)
        system_text, system_has_cache_control = _flatten_system(system)

        # Detect actual cache hits from the response (post-hoc).
        cache_read = 0
        usage = getattr(response, "usage", None)
        if usage is not None:
            cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
        prefix_was_cache_hit = cache_read > 0

        # 1. tool_descriptions — one segment for the full tools array.
        if isinstance(tools, list) and tools:
            try:
                tools_json = json.dumps(tools, sort_keys=True, default=str)
            except (TypeError, ValueError):
                tools_json = str(tools)
            if tools_has_cache_control:
                cache_state: str | None = "cache_write"
            elif prefix_was_cache_hit:
                cache_state = "cache_hit"
            else:
                cache_state = None
            segments.append(
                _make_segment(
                    position=position,
                    source="tool_descriptions",
                    content=tools_json,
                    cache_state=cache_state,
                    attrs={"tool_count": len(tools)},
                )
            )
            position += 1

        # 2. system_prompt — flatten the (possibly block-form) system field.
        if system_text:
            if system_has_cache_control:
                cache_state = "cache_write"
            elif prefix_was_cache_hit:
                cache_state = "cache_hit"
            else:
                cache_state = None
            segments.append(
                _make_segment(
                    position=position,
                    source="system_prompt",
                    content=system_text,
                    cache_state=cache_state,
                )
            )
            position += 1

        # 3. messages — walk in order. The LAST user-role message becomes
        # current_input; earlier user/assistant messages become chat_history;
        # tool_result blocks become standalone tool_observation segments.
        if isinstance(messages, list) and messages:
            # Find the index of the last user-role message that does NOT consist
            # solely of tool_result blocks (those are observations from the tool
            # loop, not the user's actual current input).
            last_user_input_idx: int | None = None
            for i in range(len(messages) - 1, -1, -1):
                msg = messages[i]
                if not isinstance(msg, dict):
                    continue
                if msg.get("role") != "user":
                    continue
                content = msg.get("content")
                _, has_tool_result = _stringify_message_content(content)
                # Pure tool_result message is an observation, not the user input.
                if has_tool_result:
                    only_tool_results = True
                    if isinstance(content, list):
                        for block in content:
                            if isinstance(block, dict) and block.get("type") not in (
                                "tool_result",
                                None,
                            ):
                                only_tool_results = False
                                break
                    else:
                        only_tool_results = False
                    if only_tool_results:
                        continue
                last_user_input_idx = i
                break

            for i, msg in enumerate(messages):
                if not isinstance(msg, dict):
                    continue
                role = msg.get("role")
                content = msg.get("content")
                records, _ = _stringify_message_content(content)
                # First, split off any tool_result blocks regardless of role —
                # they always classify as tool_observation.
                non_tool_blocks: list[dict[str, str]] = []
                for rec in records:
                    if rec["kind"] == "tool_result":
                        segments.append(
                            _make_segment(
                                position=position,
                                source="tool_observation",
                                content=rec["text"],
                                attrs={"role": role} if role else None,
                            )
                        )
                        position += 1
                    else:
                        non_tool_blocks.append(rec)

                if not non_tool_blocks:
                    continue

                combined = "\n".join(
                    rec["text"] for rec in non_tool_blocks if rec["text"]
                )
                if not combined:
                    continue

                if role == "user" and i == last_user_input_idx:
                    source = "current_input"
                else:
                    # Both earlier user messages and assistant messages
                    # (including assistant tool_use turns) are chat_history.
                    source = "chat_history"

                segments.append(
                    _make_segment(
                        position=position,
                        source=source,
                        content=combined,
                        attrs={"role": role} if role else None,
                    )
                )
                position += 1

        return segments
    except Exception as exc:  # noqa: BLE001 — never break the customer's call
        logger.debug(
            "amund: anthropic context_segments extraction swallowed error: %s",
            exc,
        )
        return []


def _extract_summary_text(response: Any) -> str:
    """Return the response's first text block, or "" if none exists.

    Anthropic's `Message.content` is a list of content blocks; for a
    summarization call, the model's textual summary lives in the first
    `{type: "text", text: ...}` block. Used by the summarize-mode hook
    to compute a `summary_content_hash` the server can later match
    against subsequent LLM spans' context_segments.

    Best-effort: returns "" on any malformed shape — never raises.
    """
    try:
        content = getattr(response, "content", None)
        if not content:
            return ""
        # Anthropic SDK returns content as list of block objects/dicts.
        for block in content:
            text = None
            # Block may be a pydantic model (has .type/.text attrs) or dict.
            if hasattr(block, "type") and getattr(block, "type", None) == "text":
                text = getattr(block, "text", None)
            elif isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text")
            if isinstance(text, str) and text:
                return text
        return ""
    except Exception:  # noqa: BLE001 — never break the recorder
        return ""


def _record_summary_memory_span(
    *,
    state: _TraceState,
    llm_span: SpanData,
    response: Any,
    start_mono: float,
) -> None:
    """Emit the sibling `memory` span linked to ``llm_span`` (MEM-11).

    Called only when summarize-mode is active (either via the
    ``summarize()`` context manager OR via ``wrap(client, summarize=True)``).
    The memory span:
      - type = "memory"
      - parent_span_id = the LLM span's span_id (= server's producer_span_id)
      - metadata.memory_op_type = "consolidate"
      - metadata.summary_token_count = response.usage.output_tokens
      - metadata.summary_content_hash = SHA-256 of the response's first text block
        (empty string -> empty-string hash, still well-defined)
    """
    try:
        usage = getattr(response, "usage", None)
        summary_tokens = 0
        if usage is not None:
            summary_tokens = int(getattr(usage, "output_tokens", 0) or 0)

        summary_text = _extract_summary_text(response)
        summary_hash = _hash_text(summary_text)

        # tokens_evicted: input context compressed away (proxy: llm_span.input_tokens)
        # tokens_injected: summary output that replaces it in the next call
        tokens_evicted = int(llm_span.input_tokens or 0)
        tokens_injected = summary_tokens

        memory_span = SpanData(
            trace_id=state.trace_id,
            type=SpanType.MEMORY,
            name="anthropic.summarize",
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
    except Exception as exc:  # noqa: BLE001 — keep recorder non-invasive
        logger.debug(
            "amund: anthropic summarize span recording swallowed error: %s", exc
        )


# ---------------------------------------------------------------------------
# MHQ-772: input_data / output_data capture for LLM spans.
#
# Every other adapter in this SDK (openai_agents, langgraph, langchain,
# claude_agent_sdk) captures prompt/response content on its LLM spans; this
# adapter previously captured none, leaving the trace-detail view materially
# less useful for the raw-Anthropic-client ICP than for every other adapter.
# The raw `openai.py` adapter has the identical gap — not fixed here, tracked
# separately, but the same helper shape should generalize to it later.
# ---------------------------------------------------------------------------


def _safe_input_messages(kwargs: dict[str, Any] | None) -> dict[str, Any] | None:
    """Build a truncated, JSON-safe input_data dict from messages.create kwargs.

    Captures ``system`` (if present) and ``messages`` as sent. Each message's
    ``content`` is truncated to ``_CONTENT_BYTE_CAP`` if it's a plain string;
    block-form content (tool_use/tool_result/image, etc.) is passed through
    as-is — those are typically small compared to a long text turn, and
    per-block truncation would require reconstructing Anthropic's content
    block shape exactly, which isn't worth the complexity for a display
    field. Returns None if there are no messages to show.
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

    result: dict[str, Any] = {"messages": safe_messages}
    system = kwargs.get("system")
    if system:
        system_text, _ = _flatten_system(system)
        if system_text:
            truncated_system, _ = _truncate(system_text)
            result["system"] = truncated_system
    return result


def _extract_output_text(response: Any) -> str | None:
    """Extract and truncate the assistant's text output from a Message response.

    Handles both TextBlock (``.text``) and ToolUseBlock (``.name``/``.input``)
    content — a response that only calls a tool still gets a meaningful
    output_data value instead of showing nothing.
    """
    content = getattr(response, "content", None)
    if not content:
        return None
    parts: list[str] = []
    for block in content:
        block_type = getattr(block, "type", None)
        if block_type == "text":
            text = getattr(block, "text", None)
            if text:
                parts.append(text)
        elif block_type == "tool_use":
            name = getattr(block, "name", "tool")
            block_input = getattr(block, "input", None)
            parts.append(
                f"[tool_use:{name}] {block_input}"
                if block_input
                else f"[tool_use:{name}]"
            )
    if not parts:
        return None
    joined = "\n".join(parts)
    truncated, _ = _truncate(joined)
    return truncated


def _record_llm_span_for_response(
    name: str,
    response: Any,
    start_mono: float,
    kwargs: dict[str, Any] | None = None,
    *,
    summarize_override: bool = False,
    started_at_iso: str | None = None,
    status: str = "completed",
    error_message: str | None = None,
    extra_metadata: dict[str, Any] | None = None,
    error: BaseException | None = None,
    span_id: str | None = None,
) -> None:
    """Build an `llm` span from the response and append it to the active trace.

    Best-effort: silent no-op if there's no active trace state. Called inside the
    monkey-patched / wrapped create() after the response is received.

    `kwargs` are the keyword arguments passed to messages.create — used to
    pre-segment the prompt into a context_segments list. Optional for callers
    that don't have them (e.g., legacy tests).

    ``summarize_override`` lets the wrapper pass its per-client default
    summarize flag down. When True, the memory-span sidecar emits regardless
    of the ``summarize()`` context manager state — the per-client choice
    sets the floor. When False, the ContextVar value decides (existing
    MEM-11 behavior preserved). The two are OR'd: ContextVar override OR
    per-client default → emit memory span.

    ``started_at_iso``: wall-clock ISO timestamp captured by the caller
    *before* the API call was made. Without this, ``SpanData.started_at``
    defaults to "now" at construction time here — i.e. *after* the call
    already completed — so the server (which derives duration_ms from the
    started_at/ended_at wall-clock pair, not the SDK's own monotonic-based
    value) would show ~0ms for every LLM span regardless of true latency.
    """
    state = get_active_trace()
    if state is None:
        return
    try:
        adapter = AnthropicAdapter()
        telemetry = adapter.extract(response)
        segments = adapter.extract_segments(kwargs or {}, response)
        _md = (kwargs or {}).get("metadata")
        apply_end_user(state, _md.get("user_id") if isinstance(_md, dict) else None)

        # ADR-023: derive per-source-bucket token composition from segments.
        total_input = telemetry.get("input_tokens") or 0
        composition = (
            compute_composition(segments or [], total_input)
            if total_input > 0
            else None
        )

        output_text = _extract_output_text(response)
        # MHQ-1880: a pre-minted id is the one a traceparent header already carried.
        span_id_kw: dict[str, Any] = {"span_id": span_id} if span_id else {}
        span = SpanData(
            trace_id=state.trace_id,
            type=SpanType.LLM,
            name=name,
            parent_span_id=state.span_stack[-1].span_id if state.span_stack else None,
            **span_id_kw,
            model=telemetry.get("model"),
            input_tokens=telemetry.get("input_tokens"),
            output_tokens=telemetry.get("output_tokens"),
            total_tokens=telemetry.get("total_tokens"),
            cost_usd=telemetry.get("cost_usd"),
            cached_input_tokens=telemetry.get("cached_input_tokens"),
            cache_write_tokens=telemetry.get("cache_write_tokens"),
            reasoning_tokens=telemetry.get("reasoning_tokens"),
            context_segments=segments if segments else None,
            input_data=_safe_input_messages(kwargs),
            output_data={"content": output_text} if output_text else None,
        )
        if composition:
            if span.metadata is None:
                span.metadata = {}
            span.metadata["context_composition"] = composition
        # MHQ-1880: the provider's message id lets the server merge this span
        # with a gateway's record of the same call instead of counting it twice.
        response_id = getattr(response, "id", None)
        if isinstance(response_id, str) and response_id:
            span.metadata = {**(span.metadata or {}), "response_id": response_id}
        if extra_metadata:
            span.metadata = {**(span.metadata or {}), **extra_metadata}

        if started_at_iso:
            span.started_at = started_at_iso
        _finalize_span(
            span, start_mono, status=status, error_message=error_message, exc=error
        )
        state.completed_spans.append(span)

        # MEM-11: emit sibling memory span when EITHER:
        #   - the `summarize()` context manager is active, OR
        #   - the wrapped client was created with summarize=True
        if _summarize_mode.get() or summarize_override:
            _record_summary_memory_span(
                state=state,
                llm_span=span,
                response=response,
                start_mono=start_mono,
            )
    except Exception as exc:  # noqa: BLE001 — keep recorder non-invasive
        logger.debug("amund: anthropic span recording swallowed error: %s", exc)


# ---------------------------------------------------------------------------
# Streaming (MHQ-1415): messages.stream() and create(stream=True)
# ---------------------------------------------------------------------------
#
# Neither path goes through Messages.create's return value: stream() posts the
# request itself, and create(stream=True) returns an event iterator. Both are
# wrapped so one llm span is recorded when the stream ends, from the finished
# message (stream()) or from usage rebuilt out of the raw events (create).

_USAGE_FIELDS = (
    "input_tokens",
    "output_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
)


def _inside_framework_llm_span() -> bool:
    """True when the innermost open span is already an ``llm`` span.

    MHQ-1415: a framework that owns the model call (LangChain's
    ``ChatAnthropic`` under ``MorseCallbackHandler``) opens its own ``llm``
    span and records the tokens from the response. The zero-code Anthropic
    patch underneath must not record the same call a second time. MHQ-133:
    the same holds when a framework marks itself the LLM owner (Strands'
    ``AnthropicModel`` while Strands traces the model call).
    """
    if get_llm_owner() is not None:
        return True
    state = get_active_trace()
    if state is None or not state.span_stack:
        return False
    return state.span_stack[-1].type == SpanType.LLM


def _record_failed_call(
    name: str,
    kwargs: dict[str, Any],
    start_mono: float,
    started_at_iso: str,
    error: BaseException,
    summarize_override: bool,
    span_id: str | None = None,
) -> None:
    """MHQ-1415: a call that raised still gets a failed ``llm`` span with the
    requested model and its latency, and no usage (so it is never priced)."""
    try:
        _record_llm_span_for_response(
            name,
            SimpleNamespace(
                model=kwargs.get("model"), usage=None, content=[], stop_reason=None
            ),
            start_mono,
            kwargs,
            summarize_override=summarize_override,
            started_at_iso=started_at_iso,
            status="failed",
            error_message=f"{type(error).__name__}: {error}",
            error=error,
            span_id=span_id,
        )
    except Exception as exc:  # noqa: BLE001 — never mask the caller's error
        logger.debug("amund: anthropic failed-call span swallowed: %s", exc)


def _traced_call_sync(
    call: Any,
    kwargs: dict[str, Any],
    *,
    summarize_override: bool,
    propagate: bool = True,
) -> Any:
    """Run one non-streaming ``messages.create`` and record its ``llm`` span.

    ``call`` takes the request kwargs (``extra_headers`` carries the
    ``traceparent`` for this call's pre-minted span id, MHQ-1880)."""
    name = "anthropic.messages.create"
    start_mono = time.monotonic()
    started_at_iso = datetime.now(UTC).isoformat()

    def _run() -> Any:
        request_kwargs, span_id = prepare_llm_call(
            kwargs, own_span=True, propagate=propagate
        )
        try:
            response = call(request_kwargs)
        except BaseException as exc:
            _record_failed_call(
                name,
                kwargs,
                start_mono,
                started_at_iso,
                exc,
                summarize_override,
                span_id,
            )
            raise
        _record_llm_span_for_response(
            name,
            response,
            start_mono,
            kwargs,
            summarize_override=summarize_override,
            started_at_iso=started_at_iso,
            span_id=span_id,
        )
        return response

    set_active_framework(_FRAMEWORK_NAME)
    try:
        if get_active_trace() is not None:
            return _run()
        # No active trace — open one for the duration of this call so the
        # span has somewhere to live.
        with _av_run(name):
            return _run()
    finally:
        clear_active_framework()


async def _traced_call_async(
    call: Any,
    kwargs: dict[str, Any],
    *,
    summarize_override: bool,
    propagate: bool = True,
) -> Any:
    """Async counterpart of ``_traced_call_sync``."""
    name = "anthropic.messages.create"
    start_mono = time.monotonic()
    started_at_iso = datetime.now(UTC).isoformat()

    async def _run() -> Any:
        request_kwargs, span_id = prepare_llm_call(
            kwargs, own_span=True, propagate=propagate
        )
        try:
            response = await call(request_kwargs)
        except BaseException as exc:
            _record_failed_call(
                name,
                kwargs,
                start_mono,
                started_at_iso,
                exc,
                summarize_override,
                span_id,
            )
            raise
        _record_llm_span_for_response(
            name,
            response,
            start_mono,
            kwargs,
            summarize_override=summarize_override,
            started_at_iso=started_at_iso,
            span_id=span_id,
        )
        return response

    set_active_framework(_FRAMEWORK_NAME)
    try:
        if get_active_trace() is not None:
            return await _run()
        with _av_run(name):
            return await _run()
    finally:
        clear_active_framework()


def _record_stream_span(
    name: str,
    message: Any,
    *,
    kwargs: dict[str, Any],
    start_mono: float,
    started_at_iso: str,
    summarize_override: bool,
    error: BaseException | None,
    span_id: str | None = None,
) -> None:
    """Record the llm span for one finished (or abandoned / failed) stream."""
    complete = message is not None and getattr(message, "stop_reason", None) is not None
    extra: dict[str, Any] = {"streamed": True}
    if error is None and not complete:
        # Consumer stopped reading before the final event: output tokens are partial.
        extra["stream_incomplete"] = True

    def _record() -> None:
        _record_llm_span_for_response(
            name,
            message,
            start_mono,
            kwargs,
            summarize_override=summarize_override,
            started_at_iso=started_at_iso,
            status="failed" if error is not None else "completed",
            error_message=f"{type(error).__name__}: {error}"
            if error is not None
            else None,
            extra_metadata=extra,
            span_id=span_id,
        )

    set_active_framework(_FRAMEWORK_NAME)
    try:
        if get_active_trace() is not None:
            _record()
        else:
            # ponytail: the auto-trace opens at record time, so its own start is the
            # stream's end; the llm span keeps the real start. Keep the trace open
            # across the stream only if trace-level timing turns out to matter.
            with _av_run(name):
                _record()
    except Exception as exc:  # noqa: BLE001 — never break the customer's stream
        logger.debug("amund: anthropic stream span recording failed: %s", exc)
    finally:
        clear_active_framework()


class _StreamSpanState:
    """Call-time bookkeeping shared by every stream wrapper; records once."""

    def __init__(
        self,
        name: str,
        kwargs: dict[str, Any],
        summarize_override: bool,
        span_id: str | None = None,
    ) -> None:
        self.span_id = span_id
        self.name = name
        self.kwargs = kwargs
        self.summarize_override = summarize_override
        self.start_mono = time.monotonic()
        self.started_at_iso = datetime.now(UTC).isoformat()
        self.done = False

    def finish(self, message: Any, error: BaseException | None) -> None:
        if self.done:
            return
        self.done = True
        if message is None and self.kwargs.get("model"):
            # Failed before the first event: keep the requested model on the span.
            message = SimpleNamespace(
                model=self.kwargs["model"], usage=None, content=[], stop_reason=None
            )
        _record_stream_span(
            self.name,
            message,
            kwargs=self.kwargs,
            start_mono=self.start_mono,
            started_at_iso=self.started_at_iso,
            summarize_override=self.summarize_override,
            error=error,
            span_id=self.span_id,
        )


def _stream_snapshot(stream: Any) -> Any:
    """The SDK's accumulated message, or None if no event arrived."""
    try:
        return stream.current_message_snapshot
    except Exception:  # noqa: BLE001 — asserts None before the first event
        return None


class _TracedMessageStreamManager:
    """Wraps ``MessageStreamManager`` (what ``messages.stream()`` returns)."""

    def __init__(
        self,
        inner: Any,
        kwargs: dict[str, Any],
        *,
        summarize_override: bool,
        span_id: str | None = None,
    ) -> None:
        self._av_inner = inner
        self._av_state = _StreamSpanState(
            "anthropic.messages.stream", kwargs, summarize_override, span_id
        )
        self._av_stream: Any = None

    def __enter__(self) -> Any:
        try:
            self._av_stream = self._av_inner.__enter__()
        except BaseException as exc:
            self._av_state.finish(None, exc)
            raise
        return self._av_stream

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> Any:
        try:
            return self._av_inner.__exit__(exc_type, exc, tb)
        finally:
            self._av_state.finish(_stream_snapshot(self._av_stream), exc)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._av_inner, name)


class _TracedAsyncMessageStreamManager:
    """Wraps ``AsyncMessageStreamManager`` (async ``messages.stream()``)."""

    def __init__(
        self,
        inner: Any,
        kwargs: dict[str, Any],
        *,
        summarize_override: bool,
        span_id: str | None = None,
    ) -> None:
        self._av_inner = inner
        self._av_state = _StreamSpanState(
            "anthropic.messages.stream", kwargs, summarize_override, span_id
        )
        self._av_stream: Any = None

    async def __aenter__(self) -> Any:
        try:
            self._av_stream = await self._av_inner.__aenter__()
        except BaseException as exc:
            self._av_state.finish(None, exc)
            raise
        return self._av_stream

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> Any:
        try:
            return await self._av_inner.__aexit__(exc_type, exc, tb)
        finally:
            self._av_state.finish(_stream_snapshot(self._av_stream), exc)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._av_inner, name)


class _RawStreamAccumulator:
    """Rebuilds model, usage and content from ``create(stream=True)`` events.

    Usage arrives in two places: ``message_start`` (input and cache tokens)
    and ``message_delta`` (cumulative output tokens). Later non-null values win.
    """

    def __init__(self) -> None:
        self.id: str | None = None
        self.model: str | None = None
        self.usage: dict[str, int] = {}
        self.stop_reason: str | None = None
        self.blocks: dict[int, dict[str, Any]] = {}

    def _merge_usage(self, usage: Any) -> None:
        for field in _USAGE_FIELDS:
            value = getattr(usage, field, None)
            if value is not None:
                self.usage[field] = value

    def feed(self, event: Any) -> None:
        etype = getattr(event, "type", None)
        if etype == "message_start":
            message = getattr(event, "message", None)
            self.id = getattr(message, "id", None)
            self.model = getattr(message, "model", None)
            self._merge_usage(getattr(message, "usage", None))
        elif etype == "message_delta":
            self._merge_usage(getattr(event, "usage", None))
            self.stop_reason = (
                getattr(getattr(event, "delta", None), "stop_reason", None)
                or self.stop_reason
            )
        elif etype == "content_block_start":
            block = getattr(event, "content_block", None)
            self.blocks[getattr(event, "index", len(self.blocks))] = {
                "type": getattr(block, "type", None),
                "name": getattr(block, "name", None),
                "parts": [],
            }
        elif etype == "content_block_delta":
            block = self.blocks.get(getattr(event, "index", -1))
            delta = getattr(event, "delta", None)
            if block is not None:
                piece = getattr(delta, "text", None) or getattr(
                    delta, "partial_json", None
                )
                if piece:
                    block["parts"].append(piece)

    def message(self) -> Any:
        if self.model is None and not self.usage:
            return None
        content = []
        for index in sorted(self.blocks):
            block = self.blocks[index]
            joined = "".join(block["parts"])
            if block["type"] == "text":
                content.append(SimpleNamespace(type="text", text=joined))
            elif block["type"] == "tool_use":
                content.append(
                    SimpleNamespace(
                        type="tool_use", name=block["name"], input=joined or None
                    )
                )
        return SimpleNamespace(
            id=self.id,
            model=self.model,
            usage=SimpleNamespace(**{f: self.usage.get(f, 0) for f in _USAGE_FIELDS}),
            content=content,
            stop_reason=self.stop_reason,
        )


class _TracedRawStream:
    """Wraps the sync ``Stream`` returned by ``messages.create(stream=True)``."""

    def __init__(
        self,
        inner: Any,
        kwargs: dict[str, Any],
        *,
        summarize_override: bool,
        span_id: str | None = None,
    ) -> None:
        self._av_inner = inner
        self._av_acc = _RawStreamAccumulator()
        self._av_state = _StreamSpanState(
            "anthropic.messages.create", kwargs, summarize_override, span_id
        )

    def __iter__(self) -> Any:
        try:
            for event in self._av_inner:
                self._av_acc.feed(event)
                yield event
        except GeneratorExit:
            # Consumer stopped early (break / close): not an error.
            self._av_state.finish(self._av_acc.message(), None)
            raise
        except BaseException as exc:
            self._av_state.finish(self._av_acc.message(), exc)
            raise
        self._av_state.finish(self._av_acc.message(), None)

    def __enter__(self) -> Any:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def close(self) -> None:
        try:
            self._av_inner.close()
        finally:
            self._av_state.finish(self._av_acc.message(), None)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._av_inner, name)


class _TracedAsyncRawStream:
    """Wraps the ``AsyncStream`` from async ``messages.create(stream=True)``."""

    def __init__(
        self,
        inner: Any,
        kwargs: dict[str, Any],
        *,
        summarize_override: bool,
        span_id: str | None = None,
    ) -> None:
        self._av_inner = inner
        self._av_acc = _RawStreamAccumulator()
        self._av_state = _StreamSpanState(
            "anthropic.messages.create", kwargs, summarize_override, span_id
        )

    async def __aiter__(self) -> Any:
        try:
            async for event in self._av_inner:
                self._av_acc.feed(event)
                yield event
        except GeneratorExit:
            self._av_state.finish(self._av_acc.message(), None)
            raise
        except BaseException as exc:
            self._av_state.finish(self._av_acc.message(), exc)
            raise
        self._av_state.finish(self._av_acc.message(), None)

    async def __aenter__(self) -> Any:
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.close()

    async def close(self) -> None:
        try:
            await self._av_inner.close()
        finally:
            self._av_state.finish(self._av_acc.message(), None)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._av_inner, name)


# ---------------------------------------------------------------------------
# wrap() — recommended public API (mirrors TypeScript wrapAnthropic)
# ---------------------------------------------------------------------------


class _WrappedMessages:
    """Drop-in replacement for ``client.messages``.

    Holds a reference to the underlying ``messages`` resource and intercepts
    ``create`` (sync + async) to emit Amund spans. Every other attribute
    is delegated via ``__getattr__`` so streaming helpers
    (``messages.stream``, ``messages.batches``, etc.) keep working unchanged.
    """

    def __init__(
        self, inner: Any, *, summarize_default: bool, propagate: bool = True
    ) -> None:
        # ``_av_inner`` and ``_av_summarize_default`` use the ``_av_`` prefix
        # to avoid colliding with any attribute the upstream Anthropic SDK
        # may add in future versions.
        self._av_inner = inner
        self._av_summarize_default = summarize_default
        self._av_propagate = propagate

    def create(self, *args: Any, **kwargs: Any) -> Any:
        """Sync wrapper around ``messages.create`` — emits an ``llm`` span.

        AGE-240: runtime dedup vs ``install()``. If ``install()`` has
        monkey-patched ``Messages.create`` globally (which auto-detect
        does on first SDK init), the inner call ALREADY emits a span via
        the global patch. We must NOT emit a second one here or every
        call gets recorded twice. This is detected at call-time (not
        wrap-time) because customer code may call ``wrap()`` before any
        SDK code runs.
        """
        if _install_active():
            # The global patch records it (and injects the traceparent).
            return self._av_inner.create(*args, **kwargs)
        if _inside_framework_llm_span():
            # The enclosing framework span records it; MHQ-1880: point the
            # traceparent at that span.
            kw, _ = prepare_llm_call(
                kwargs, own_span=False, propagate=self._av_propagate
            )
            return self._av_inner.create(*args, **kw)
        if kwargs.get("stream"):
            kw, span_id = prepare_llm_call(
                kwargs, own_span=True, propagate=self._av_propagate
            )
            return _TracedRawStream(
                self._av_inner.create(*args, **kw),
                kwargs,
                summarize_override=self._av_summarize_default,
                span_id=span_id,
            )
        return _traced_call_sync(
            lambda kw: self._av_inner.create(*args, **kw),
            kwargs,
            summarize_override=self._av_summarize_default,
            propagate=self._av_propagate,
        )

    def stream(self, *args: Any, **kwargs: Any) -> Any:
        """``messages.stream``: one ``llm`` span when the block ends (MHQ-1415)."""
        if _install_active():
            return self._av_inner.stream(*args, **kwargs)
        if _inside_framework_llm_span():
            kw, _ = prepare_llm_call(
                kwargs, own_span=False, propagate=self._av_propagate
            )
            return self._av_inner.stream(*args, **kw)
        kw, span_id = prepare_llm_call(
            kwargs, own_span=True, propagate=self._av_propagate
        )
        return _TracedMessageStreamManager(
            self._av_inner.stream(*args, **kw),
            kwargs,
            summarize_override=self._av_summarize_default,
            span_id=span_id,
        )

    def __getattr__(self, name: str) -> Any:
        """Delegate every other attribute/method to the underlying messages."""
        # ``__getattr__`` only runs for misses, so this never shadows
        # ``create`` (defined above) or the ``_av_*`` attrs (set in __init__
        # and present in __dict__).
        return getattr(self._av_inner, name)


class _WrappedAsyncMessages:
    """Async counterpart of ``_WrappedMessages`` for ``AsyncAnthropic`` clients."""

    def __init__(
        self, inner: Any, *, summarize_default: bool, propagate: bool = True
    ) -> None:
        self._av_inner = inner
        self._av_summarize_default = summarize_default
        self._av_propagate = propagate

    async def create(self, *args: Any, **kwargs: Any) -> Any:
        if _install_active():
            return await self._av_inner.create(*args, **kwargs)
        if _inside_framework_llm_span():
            kw, _ = prepare_llm_call(
                kwargs, own_span=False, propagate=self._av_propagate
            )
            return await self._av_inner.create(*args, **kw)
        if kwargs.get("stream"):
            kw, span_id = prepare_llm_call(
                kwargs, own_span=True, propagate=self._av_propagate
            )
            return _TracedAsyncRawStream(
                await self._av_inner.create(*args, **kw),
                kwargs,
                summarize_override=self._av_summarize_default,
                span_id=span_id,
            )
        return await _traced_call_async(
            lambda kw: self._av_inner.create(*args, **kw),
            kwargs,
            summarize_override=self._av_summarize_default,
            propagate=self._av_propagate,
        )

    def stream(self, *args: Any, **kwargs: Any) -> Any:
        """Async ``messages.stream`` — returns a manager used with ``async with``."""
        if _install_active():
            return self._av_inner.stream(*args, **kwargs)
        if _inside_framework_llm_span():
            kw, _ = prepare_llm_call(
                kwargs, own_span=False, propagate=self._av_propagate
            )
            return self._av_inner.stream(*args, **kw)
        kw, span_id = prepare_llm_call(
            kwargs, own_span=True, propagate=self._av_propagate
        )
        return _TracedAsyncMessageStreamManager(
            self._av_inner.stream(*args, **kw),
            kwargs,
            summarize_override=self._av_summarize_default,
            span_id=span_id,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._av_inner, name)


class _WrappedClient:
    """Delegating wrapper around an Anthropic (or AsyncAnthropic) client.

    Returned by ``wrap()``. Replaces ``client.messages`` with a
    ``_WrappedMessages`` (or ``_WrappedAsyncMessages``) and delegates
    every other attribute to the underlying client via ``__getattr__``.

    The wrapper is intentionally a plain class rather than a Proxy or a
    subclass of ``Anthropic`` — it requires no import of the anthropic
    package at module load time (the adapter ships as an extra) and works
    against any structural client that exposes ``.messages.create``.
    """

    def __init__(self, inner: Any, *, summarize: bool, propagate: bool = True) -> None:
        self._av_inner = inner
        self._av_summarize_default = summarize
        # Detect async vs sync. AsyncAnthropic.messages.create is a coroutine
        # function; Anthropic.messages.create is plain. Inspect the inner
        # client class name as a cheap, dep-free hint; fall back to checking
        # the create attribute itself.
        is_async = self._detect_async(inner)
        inner_messages = getattr(inner, "messages", None)
        if inner_messages is None:
            # Defensive: not a real client. Wrap nothing — calls will fail
            # the same way they would on the unwrapped instance.
            self._av_messages: Any = None
        elif is_async:
            self._av_messages = _WrappedAsyncMessages(
                inner_messages, summarize_default=summarize, propagate=propagate
            )
        else:
            self._av_messages = _WrappedMessages(
                inner_messages, summarize_default=summarize, propagate=propagate
            )
        # Idempotency marker — wrap()-ing an already-wrapped client should
        # be a no-op; we check this attribute in ``wrap()`` itself.
        setattr(self, _WRAPPED_MARKER, True)

    @staticmethod
    def _detect_async(inner: Any) -> bool:
        # Strategy 1: class name. Anthropic SDK uses AsyncAnthropic /
        # AsyncMessages for the async variants.
        cls_name = type(inner).__name__
        if cls_name.startswith("Async"):
            return True
        # Strategy 2: inspect the create method.
        messages = getattr(inner, "messages", None)
        if messages is None:
            return False
        create = getattr(messages, "create", None)
        # ``inspect.iscoroutinefunction`` would also work, but a
        # dep-free duck check keeps the import surface minimal.
        return getattr(create, "__code__", None) is not None and bool(
            create.__code__.co_flags & 0x80  # CO_COROUTINE
        )

    @property
    def messages(self) -> Any:
        """Return the wrapped ``messages`` resource."""
        return self._av_messages

    def __getattr__(self, name: str) -> Any:
        """Delegate every other attribute to the underlying client.

        ``__getattr__`` only fires on misses, so ``messages`` (property),
        ``_av_*`` (set in __init__), and ``_WRAPPED_MARKER`` are served
        from __dict__/the class without touching the inner client.
        """
        return getattr(self._av_inner, name)


def wrap(
    client: ClientT,
    *,
    summarize: bool = False,
    propagate_trace_context: bool = True,
) -> ClientT:
    """Wrap an Anthropic client so every ``messages.create`` call emits AV spans.

    Mirrors the TypeScript SDK's ``wrapAnthropic(client, options)``. The
    returned object is a drop-in replacement for the original client: every
    attribute except ``messages`` is delegated via ``__getattr__``, and
    ``messages.create`` is replaced by an instrumented version that records
    an ``llm`` span with pre-segmented ``context_segments`` (CTX-08).

    Args:
        client: An ``anthropic.Anthropic`` or ``anthropic.AsyncAnthropic``
            instance (or any structural look-alike — only ``client.messages
            .create`` is required).
        summarize: When ``True``, every wrapped call also emits a sibling
            ``memory`` span (``metadata.memory_op_type="consolidate"``) per
            MEM-11. The per-call ``summarize()`` context manager remains
            available as an override that wins when active.
        propagate_trace_context: When ``True`` (default), each call carries a
            W3C ``traceparent`` request header naming this call's ``llm`` span,
            so a gateway that forwards it joins the same trace (MHQ-1880).
            ``AMUND_PROPAGATE_TRACE_CONTEXT=false`` also turns it off.

    Returns:
        The wrapped client. The return type is the same as the input client
        (typed via TypeVar so callers see ``wrap(Anthropic()) -> Anthropic``).
        Calling ``wrap()`` on an already-wrapped client is idempotent —
        returns the same instance.

    Example::

        from amund.adapters import anthropic as av_anthropic
        import anthropic as anthropic_sdk

        client = av_anthropic.wrap(anthropic_sdk.Anthropic())

        # Every call is traced; no further Amund code required.
        response = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=100,
            messages=[{"role": "user", "content": "hello"}],
        )

    Recommended over ``install()`` (which monkey-patches the Anthropic class
    globally — useful for zero-config auto-detect, but heavier-handed).
    """
    if client is None:
        return client
    # Idempotency: a previously wrapped client is returned as-is.
    if getattr(client, _WRAPPED_MARKER, False):
        return client
    # AGE-240 dogfood fix: when the module-level ``install()`` has already
    # monkey-patched ``Messages.create`` globally (which auto-detect does on
    # first SDK use), wrapping per-instance would record every call TWICE —
    # once via the global patch + once via the wrapper. Return the client
    # unmodified so the global patch is the sole recorder. The wrapper still
    # exists for customers who deliberately call ``wrap()`` without auto-
    # detect / install() (e.g., advanced setups that prefer per-client
    # control).
    try:
        import anthropic  # type: ignore[import-not-found]

        if getattr(anthropic, _INSTALL_FLAG, False):
            return client
    except ImportError:
        pass
    try:
        # ``_WrappedClient`` is intentionally untyped against ``ClientT``
        # because its public surface is the *same* as the inner client (via
        # __getattr__). The cast keeps static type-checkers happy.
        return _WrappedClient(  # type: ignore[return-value]
            client, summarize=summarize, propagate=propagate_trace_context
        )
    except Exception as exc:  # noqa: BLE001 — never break the customer's setup
        logger.warning(
            "amund: failed to wrap Anthropic client (%s) — calls untraced",
            exc,
        )
        return client


# ---------------------------------------------------------------------------
# install() / uninstall() — module-level monkey-patch (auto-detect compat)
# ---------------------------------------------------------------------------


def _wrap_create(original):
    """Build the sync replacement for Messages.create (install-style)."""

    def traced_create(self, *args, **kwargs):
        if _inside_framework_llm_span():
            # MHQ-1880: with_raw_response binds this patched method, so a
            # LangChain call lands here; point the traceparent at its llm span.
            kw, _ = prepare_llm_call(kwargs, own_span=False)
            return original(self, *args, **kw)
        if kwargs.get("stream"):
            kw, span_id = prepare_llm_call(kwargs, own_span=True)
            return _TracedRawStream(
                original(self, *args, **kw),
                kwargs,
                summarize_override=False,
                span_id=span_id,
            )
        return _traced_call_sync(
            lambda kw: original(self, *args, **kw), kwargs, summarize_override=False
        )

    traced_create.__wrapped__ = original  # standard convention so customers can unwrap
    return traced_create


def _wrap_stream(original, manager_cls):
    """Build the replacement for (Async)Messages.stream (install-style)."""

    def traced_stream(self, *args, **kwargs):
        if _inside_framework_llm_span():
            kw, _ = prepare_llm_call(kwargs, own_span=False)
            return original(self, *args, **kw)
        kw, span_id = prepare_llm_call(kwargs, own_span=True)
        return manager_cls(
            original(self, *args, **kw),
            kwargs,
            summarize_override=False,
            span_id=span_id,
        )

    traced_stream.__wrapped__ = original
    return traced_stream


def _wrap_acreate(original):
    """Build the async replacement for AsyncMessages.create (install-style)."""

    async def traced_acreate(self, *args, **kwargs):
        if _inside_framework_llm_span():
            kw, _ = prepare_llm_call(kwargs, own_span=False)
            return await original(self, *args, **kw)
        if kwargs.get("stream"):
            kw, span_id = prepare_llm_call(kwargs, own_span=True)
            return _TracedAsyncRawStream(
                await original(self, *args, **kw),
                kwargs,
                summarize_override=False,
                span_id=span_id,
            )
        return await _traced_call_async(
            lambda kw: original(self, *args, **kw), kwargs, summarize_override=False
        )

    traced_acreate.__wrapped__ = original
    return traced_acreate


def is_installed() -> bool:
    """Return True if the Anthropic adapter has installed its monkey-patches.

    Only reports on the module-level ``install()`` patch — ``wrap()``-style
    instrumentation is per-instance and not tracked globally.
    """
    try:
        import anthropic  # type: ignore[import-not-found]
    except ImportError:
        return False
    return getattr(anthropic, _INSTALL_FLAG, False)


@absorb_errors(fingerprint="morse_ai.adapters.anthropic.install")
def install() -> bool:
    """Monkey-patch the Anthropic SDK class so every client is instrumented.

    Auto-detect compatibility shim — called by
    ``amund._internal.auto_detect.install_adapters()`` so that
    ``pip install amund`` + ``AMUND_API_KEY=...`` is a complete
    setup without further code changes. For new code, prefer ``wrap(client)``
    which is per-instance and avoids touching global state.

    Idempotent — safe to call repeatedly. Returns True if installed (newly
    or already), False if Anthropic isn't available or installation failed.
    """
    global _original_create, _original_acreate, _original_stream, _original_astream
    global _original_beta_create, _original_beta_acreate

    try:
        import anthropic  # type: ignore[import-not-found]
    except ImportError:
        return False

    if getattr(anthropic, _INSTALL_FLAG, False):
        return True

    try:
        from anthropic.resources.messages import (
            Messages,  # type: ignore[import-not-found]
        )

        _original_create = Messages.create
        Messages.create = _wrap_create(_original_create)  # type: ignore[method-assign]
    except (ImportError, AttributeError) as exc:
        logger.warning(
            "amund: Anthropic adapter could not patch Messages.create: %s — "
            "Anthropic calls will run untraced",
            exc,
        )
        return False

    try:
        from anthropic.resources.messages import (
            AsyncMessages,  # type: ignore[import-not-found]
        )

        _original_acreate = AsyncMessages.create
        AsyncMessages.create = _wrap_acreate(_original_acreate)  # type: ignore[method-assign]
    except (ImportError, AttributeError):
        # Async path is optional — older anthropic versions may not expose it.
        pass

    # MHQ-1415: stream() posts its own request and never calls create().
    try:
        from anthropic.resources.messages import (
            AsyncMessages,  # type: ignore[import-not-found]
            Messages,  # type: ignore[import-not-found]
        )

        _original_stream = Messages.stream
        Messages.stream = _wrap_stream(_original_stream, _TracedMessageStreamManager)  # type: ignore[method-assign]
        _original_astream = AsyncMessages.stream
        AsyncMessages.stream = _wrap_stream(  # type: ignore[method-assign]
            _original_astream, _TracedAsyncMessageStreamManager
        )
    except (ImportError, AttributeError) as exc:
        logger.debug(
            "amund: Anthropic adapter could not patch messages.stream: %s", exc
        )

    # MHQ-1880: ``client.beta.messages.create`` (LangChain uses it when
    # ``betas`` is set) is a separate class; patch it the same way so those
    # calls are traced and carry the traceparent.
    try:
        from anthropic.resources.beta.messages import (  # type: ignore[import-not-found]
            AsyncMessages as BetaAsyncMessages,
        )
        from anthropic.resources.beta.messages import (  # type: ignore[import-not-found]
            Messages as BetaMessages,
        )

        _original_beta_create = BetaMessages.create
        BetaMessages.create = _wrap_create(_original_beta_create)  # type: ignore[method-assign]
        _original_beta_acreate = BetaAsyncMessages.create
        BetaAsyncMessages.create = _wrap_acreate(_original_beta_acreate)  # type: ignore[method-assign]
    except (ImportError, AttributeError) as exc:
        logger.debug(
            "amund: Anthropic adapter could not patch beta.messages.create: %s",
            exc,
        )

    setattr(anthropic, _INSTALL_FLAG, True)
    logger.info("amund: Anthropic adapter installed")
    return True


@absorb_errors(fingerprint="morse_ai.adapters.anthropic.uninstall")
def uninstall() -> None:
    """Restore the original Anthropic methods. Idempotent."""
    global _original_create, _original_acreate, _original_stream, _original_astream
    global _original_beta_create, _original_beta_acreate

    try:
        import anthropic  # type: ignore[import-not-found]
    except ImportError:
        return

    if not getattr(anthropic, _INSTALL_FLAG, False):
        return

    if _original_create is not None:
        try:
            from anthropic.resources.messages import (
                Messages,  # type: ignore[import-not-found]
            )

            Messages.create = _original_create  # type: ignore[method-assign]
        except (ImportError, AttributeError):
            pass
        _original_create = None

    if _original_acreate is not None:
        try:
            from anthropic.resources.messages import (
                AsyncMessages,  # type: ignore[import-not-found]
            )

            AsyncMessages.create = _original_acreate  # type: ignore[method-assign]
        except (ImportError, AttributeError):
            pass
        _original_acreate = None

    if _original_stream is not None or _original_astream is not None:
        try:
            from anthropic.resources.messages import (
                AsyncMessages,  # type: ignore[import-not-found]
                Messages,  # type: ignore[import-not-found]
            )

            if _original_stream is not None:
                Messages.stream = _original_stream  # type: ignore[method-assign]
            if _original_astream is not None:
                AsyncMessages.stream = _original_astream  # type: ignore[method-assign]
        except (ImportError, AttributeError):
            pass
        _original_stream = None
        _original_astream = None

    if _original_beta_create is not None or _original_beta_acreate is not None:
        try:
            from anthropic.resources.beta.messages import (  # type: ignore[import-not-found]
                AsyncMessages as BetaAsyncMessages,
            )
            from anthropic.resources.beta.messages import (  # type: ignore[import-not-found]
                Messages as BetaMessages,
            )

            if _original_beta_create is not None:
                BetaMessages.create = _original_beta_create  # type: ignore[method-assign]
            if _original_beta_acreate is not None:
                BetaAsyncMessages.create = _original_beta_acreate  # type: ignore[method-assign]
        except (ImportError, AttributeError):
            pass
        _original_beta_create = None
        _original_beta_acreate = None

    setattr(anthropic, _INSTALL_FLAG, False)
