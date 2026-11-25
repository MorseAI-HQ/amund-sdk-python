"""Context segment decomposer for OpenAI-compatible message arrays.

Pure functions — no I/O, no imports from the rest of AV. Called from any
adapter that intercepts an LLM call in the OpenAI chat completions format
(which covers OpenAI, LangChain, and LangGraph after message conversion).

The Anthropic adapter uses its own ``_extract_anthropic_segments`` because
Anthropic's message format has extra structure (block lists, tool_result
blocks, cache_control markers). For OpenAI-format messages this module is
the canonical decomposer.

Source tag mapping (mirrors the backend ``SegmentSource`` enum):

  role=system                                      → system_prompt
  tools=[...]  (the definitions array)             → tool_descriptions
  role=tool (OpenAI tool result)                   → tool_observation
  role=user / role=assistant (prior turns)         → chat_history
  role=user (last user turn, not a tool result)    → current_input

Position is 0-indexed in emission order:
  tool_descriptions → system_prompt → chat_history/tool_observation → current_input
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

logger = logging.getLogger("amund")

# Per-segment content cap (bytes). Matches the Anthropic adapter's cap so
# the server-side payload guard (AI-SEC-08, 256 KB on ContextSegmentIn.content)
# is approached at the same rate for all adapters.
_CONTENT_BYTE_CAP = 4096


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _truncate(text: str) -> tuple[str, bool]:
    """Truncate to _CONTENT_BYTE_CAP bytes, preserving a valid UTF-8 prefix."""
    encoded = text.encode("utf-8", errors="replace")
    if len(encoded) <= _CONTENT_BYTE_CAP:
        return text, False
    truncated = encoded[:_CONTENT_BYTE_CAP].decode("utf-8", errors="ignore")
    return truncated, True


def _make_segment(
    *,
    position: int,
    source: str,
    content: str,
    attrs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    truncated_text, was_truncated = _truncate(content)
    seg: dict[str, Any] = {
        "position": position,
        "source": source,
        "content_hash": _hash_text(content),
        "content": truncated_text,
        "content_truncated": was_truncated,
        "tokens": None,
        "tokens_estimated": False,
    }
    if attrs:
        seg["attrs"] = attrs
    return seg


def _content_to_text(content: Any) -> str:
    """Coerce an OpenAI message content value to a plain string.

    OpenAI content can be:
      - str: plain text (most common)
      - list of parts: [{type: "text", text: "..."}, {type: "image_url", ...}]
      - None
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text":
                    parts.append(str(part.get("text", "")))
                elif part.get("type") == "image_url":
                    parts.append("[image]")
            elif isinstance(part, str):
                parts.append(part)
        return "\n".join(p for p in parts if p)
    return str(content)


def decompose_openai_messages(
    messages: list[Any] | None,
    tools: list[Any] | None = None,
) -> list[dict[str, Any]]:
    """Decompose an OpenAI chat completions request into context segments.

    Args:
        messages: The ``messages`` parameter from ``chat.completions.create``.
            Each element should be a dict with ``role`` and ``content`` keys,
            or an object with those attributes.
        tools: The ``tools`` parameter (list of tool definition dicts).

    Returns:
        List of context segment dicts ready for the AV ingest endpoint.
        Returns [] (never raises) on malformed input.
    """
    if not messages and not tools:
        return []

    try:
        segments: list[dict[str, Any]] = []
        position = 0

        # 1. tool_descriptions — one segment for the entire tools array.
        if isinstance(tools, list) and tools:
            try:
                tool_names = [
                    t.get("function", {}).get("name") or t.get("name", "")
                    for t in tools
                    if isinstance(t, dict)
                ]
                tools_json = json.dumps(tools, sort_keys=True, default=str)
            except (TypeError, ValueError):
                tool_names = []
                tools_json = str(tools)
            segments.append(
                _make_segment(
                    position=position,
                    source="tool_descriptions",
                    content=tools_json,
                    attrs={
                        "tool_count": len(tools),
                        "tool_names": [n for n in tool_names if n],
                    },
                )
            )
            position += 1

        if not isinstance(messages, list):
            return segments

        # 2. system_prompt — the system message (there's usually at most one).
        system_parts: list[str] = []
        for msg in messages:
            role = _get_role(msg)
            if role == "system":
                text = _content_to_text(_get_content(msg))
                if text:
                    system_parts.append(text)

        if system_parts:
            segments.append(
                _make_segment(
                    position=position,
                    source="system_prompt",
                    content="\n".join(system_parts),
                )
            )
            position += 1

        # 3. Walk non-system messages. Find the last user message that is NOT
        #    purely a tool_result (role=tool or role=user with tool call outputs).
        non_system = [m for m in messages if _get_role(m) != "system"]

        last_user_input_idx: int | None = None
        for i in range(len(non_system) - 1, -1, -1):
            msg = non_system[i]
            role = _get_role(msg)
            if role == "user":
                last_user_input_idx = i
                break

        for i, msg in enumerate(non_system):
            role = _get_role(msg)
            content = _get_content(msg)

            # OpenAI tool role → tool_observation segment.
            if role == "tool":
                text = _content_to_text(content)
                if text:
                    tool_call_id = _get_attr(msg, "tool_call_id") or ""
                    prefix = (
                        f"[tool_result id={tool_call_id}]\n" if tool_call_id else ""
                    )
                    segments.append(
                        _make_segment(
                            position=position,
                            source="tool_observation",
                            content=prefix + text,
                            attrs={"role": role},
                        )
                    )
                    position += 1
                continue

            # Skip system (already handled).
            if role == "system":
                continue

            text = _content_to_text(content)
            if not text:
                continue

            if role == "user" and i == last_user_input_idx:
                source = "current_input"
            else:
                source = "chat_history"

            segments.append(
                _make_segment(
                    position=position,
                    source=source,
                    content=text,
                    attrs={"role": role} if role else None,
                )
            )
            position += 1

        return segments

    except Exception as exc:  # noqa: BLE001 — never break customer code
        logger.debug("amund: context decomposer swallowed error: %s", exc)
        return []


def _get_role(msg: Any) -> str:
    """Extract the ``role`` field from a message dict or object."""
    if isinstance(msg, dict):
        return str(msg.get("role", ""))
    # LangChain BaseMessage: .type == "human" | "system" | "ai" | "tool"
    lc_type = getattr(msg, "type", None)
    if lc_type is not None:
        return _lc_type_to_role(str(lc_type))
    return str(getattr(msg, "role", ""))


def _get_content(msg: Any) -> Any:
    if isinstance(msg, dict):
        return msg.get("content")
    return getattr(msg, "content", None)


def _get_attr(msg: Any, attr: str) -> Any:
    if isinstance(msg, dict):
        return msg.get(attr)
    return getattr(msg, attr, None)


def _lc_type_to_role(lc_type: str) -> str:
    """Map LangChain message type to OpenAI role string."""
    mapping = {
        "human": "user",
        "ai": "assistant",
        "system": "system",
        "tool": "tool",
        "function": "tool",
        "chat": "user",
    }
    return mapping.get(lc_type.lower(), lc_type)


# Source → composition bucket
_SOURCE_TO_BUCKET: dict[str, str] = {
    "system_prompt": "system",
    "chat_history": "history",
    "current_input": "current_input",
    "tool_descriptions": "tools",
    "tool_observation": "tools",
    "injected_memory": "memory",
    "retrieved_context": "retrieval",
    "retrieval": "retrieval",
}

_COMPOSITION_KEYS = (
    "system",
    "history",
    "memory",
    "retrieval",
    "tools",
    "current_input",
)


def compute_composition(
    segments: list[dict],
    total_input_tokens: int,
) -> dict[str, int]:
    """Return per-bucket token counts derived from context_segments.

    When a segment has an explicit ``tokens`` count, that value is used.
    Otherwise, character length is used as a proxy for proportional
    distribution of ``total_input_tokens`` across segments without counts.
    The sum of all bucket values equals ``total_input_tokens``.

    Returns a dict with keys: system, history, memory, retrieval, tools,
    current_input — all int, all >= 0.
    """
    result: dict[str, int] = {k: 0 for k in _COMPOSITION_KEYS}
    if not segments or total_input_tokens <= 0:
        return result

    # Separate segments with known token counts from those needing estimation.
    known_total = 0
    known: list[tuple[str, int]] = []
    unknown: list[tuple[str, int]] = []  # (bucket, char_len)

    for seg in segments:
        bucket = _SOURCE_TO_BUCKET.get(seg.get("source", ""), "history")
        tok = seg.get("tokens")
        if isinstance(tok, int) and tok > 0:
            known.append((bucket, tok))
            known_total += tok
        else:
            char_len = len(seg.get("content") or "")
            unknown.append((bucket, char_len))

    # Attribute known-token segments first.
    for bucket, tok in known:
        result[bucket] += tok

    # Distribute remaining tokens among unknown segments by char proportion.
    remaining = total_input_tokens - known_total
    if remaining > 0 and unknown:
        total_chars = sum(c for _, c in unknown)
        if total_chars > 0:
            distributed = 0
            for i, (bucket, chars) in enumerate(unknown):
                if i == len(unknown) - 1:
                    # Last segment gets the remainder to avoid rounding drift.
                    alloc = remaining - distributed
                else:
                    alloc = round(remaining * chars / total_chars)
                result[bucket] += alloc
                distributed += alloc

    return result
