"""Shared builder for the retriever ``documents[]`` wire field (@beta, RAGQ-01).

Each retrieved chunk is reported as::

    {"rank": 1, "id": "...", "content": "<=2KB", "score": 0.9, "source": "..."}

plus ``"truncated": true`` when the content was cut at the cap. At most
``MAX_DOCUMENTS`` chunks are reported. The field is additive: legacy keys
(``content_preview``, ``docs_count``, ``sources``...) stay on the span, and a
server that does not know ``documents[].content`` simply ignores it.
"""

from __future__ import annotations

import hashlib
from typing import Any

#: Max chunks reported per retriever span.
MAX_DOCUMENTS = 10
#: Max UTF-8 bytes of chunk content reported per chunk.
MAX_CONTENT_BYTES = 2048


def _truncate(content: str) -> tuple[str, bool]:
    encoded = content.encode("utf-8")
    if len(encoded) <= MAX_CONTENT_BYTES:
        return content, False
    return encoded[:MAX_CONTENT_BYTES].decode("utf-8", errors="ignore"), True


def _as_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return "" if value is None else str(value)


def _float_or_none(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_documents(docs: Any) -> list[dict[str, Any]]:
    """Normalise retrieved documents into the ``documents[]`` wire shape.

    Accepts LangChain ``Document`` objects (``page_content`` + ``metadata``)
    and plain dicts. Never raises; unreadable entries are skipped.
    """
    out: list[dict[str, Any]] = []
    try:
        seq = list(docs or [])[:MAX_DOCUMENTS]
    except Exception:  # noqa: BLE001
        return out
    for doc in seq:
        try:
            if isinstance(doc, dict):
                content = _as_text(doc.get("page_content", doc.get("content")))
                meta = doc.get("metadata")
                meta = meta if isinstance(meta, dict) else {}
                raw_id = doc.get("id", meta.get("id"))
                score = doc.get("score", meta.get("score", meta.get("relevance_score")))
                source = doc.get("source", meta.get("source"))
            else:
                content = _as_text(getattr(doc, "page_content", None))
                meta = getattr(doc, "metadata", None)
                meta = meta if isinstance(meta, dict) else {}
                raw_id = getattr(doc, "id", None) or meta.get("id")
                score = meta.get("score", meta.get("relevance_score"))
                source = meta.get("source")
            text, truncated = _truncate(content)
            if raw_id is None or raw_id == "":
                raw_id = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
            entry: dict[str, Any] = {
                "rank": len(out) + 1,
                "id": str(raw_id),
                "content": text,
                "score": _float_or_none(score),
                "source": _as_text(source),
            }
            if truncated:
                entry["truncated"] = True
            out.append(entry)
        except Exception:  # noqa: BLE001 — telemetry must never break the app
            continue
    return out
