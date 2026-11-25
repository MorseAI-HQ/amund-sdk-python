"""Shared ``json.dumps(default=...)`` fallback for non-primitive values.

Used by both the legacy HTTP transport (``transport.py``) and the OTel
engine (``otel_provider.json_capped``) so span ``input_data`` /
``output_data`` serialize the same way on both paths.
"""

from __future__ import annotations

from typing import Any

# Always kept on dumped messages, even when empty (an AI tool-call turn has
# ``content == ""``): the server's messages classifier keys off these.
_MESSAGE_REQUIRED_KEYS = frozenset({"type", "content"})


def _is_empty(value: Any) -> bool:
    # ``0`` and ``False`` are meaningful and kept; only None and empty
    # strings / containers are dropped.
    return value is None or (
        isinstance(value, (str, list, dict, tuple)) and len(value) == 0
    )


def _is_message_like(obj: Any) -> bool:
    """True for LangChain/LangGraph ``BaseMessage``-shaped objects.

    Duck-typed (no langchain import): a pydantic model exposing both a
    ``type`` (``"human"``, ``"ai"``, ``"tool"``, ...) and a ``content``
    attribute.
    """
    return (
        callable(getattr(obj, "model_dump", None))
        and isinstance(getattr(obj, "type", None), str)
        and hasattr(obj, "content")
    )


def json_default(obj: Any) -> Any:
    """Fallback JSON serializer for non-primitive values.

    AGE-240: LangChain / LangGraph callback handlers receive ``inputs`` and
    ``response`` payloads that contain custom objects (HumanMessage,
    AIMessage, Document, ndarray, datetime, Decimal, UUID, etc.). These
    leak into span ``input_data`` / ``output_data`` and break the default
    ``json.dumps`` with ``TypeError: Object of type X is not JSON
    serializable``, dropping the whole batch.

    MHQ-1936: message objects must serialize as structured dicts (``type``,
    ``content``, ``tool_calls``, ...), never as their Python repr, or the
    server's messages classifier loses the role. Empty top-level fields
    (``None``, ``""``, ``[]``, ``{}``) are dropped from dumped messages to
    keep payloads small; ``type`` and ``content`` are always kept.

    Strategy (cheap -> expensive):
    1. Pydantic-like model: ``model_dump`` or ``dict()`` returns a dict.
    2. Generic dataclass / NamedTuple / hasattr ``__dict__``: vars().
    3. iterable (set, generator, ndarray-like): list().
    4. final fallback: ``str(obj)`` — guarantees a serializable result.

    Never raises — at worst returns ``"<unserializable>"``.
    """
    try:
        # 1) Pydantic v2 -> v1 -> other model libs
        dump = getattr(obj, "model_dump", None) or getattr(obj, "dict", None)
        if callable(dump):
            try:
                dumped = dump()
            except Exception:  # noqa: BLE001
                pass
            else:
                if isinstance(dumped, dict) and _is_message_like(obj):
                    return {
                        k: v
                        for k, v in dumped.items()
                        if k in _MESSAGE_REQUIRED_KEYS or not _is_empty(v)
                    }
                return dumped
        # 2) dataclasses / generic objects
        if hasattr(obj, "__dict__"):
            try:
                return {k: v for k, v in vars(obj).items() if not k.startswith("_")}
            except Exception:  # noqa: BLE001
                pass
        # 3) iterable
        if hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes)):
            try:
                return list(obj)
            except Exception:  # noqa: BLE001
                pass
        # 4) final fallback
        return str(obj)
    except Exception:  # noqa: BLE001
        return "<unserializable>"
