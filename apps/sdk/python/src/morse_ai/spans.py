"""SpanData — typed span model for the Morse SDK v2 wire format."""

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any


class SpanType(StrEnum):
    """Span type values — must match backend SpanType.TextChoices exactly."""

    AGENT = "agent"
    LLM = "llm"
    TOOL = "tool"
    HTTP = "http"
    DB = "db"
    CUSTOM = "custom"
    MEMORY = "memory"
    RETRIEVER = "retriever"
    GUARDRAIL = "guardrail"
    HANDOFF = "handoff"


@dataclass
class SpanData:
    """A single span in a distributed trace.

    Maps 1:1 to the backend Span model fields. The wire format is
    SpanData.to_dict() embedded inside a {"traces": [...]} payload.
    """

    trace_id: str
    type: SpanType
    name: str

    # Auto-generated identifiers
    span_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    started_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    # Hierarchy
    parent_span_id: str | None = None

    # Timing (filled in on span close)
    ended_at: str | None = None
    duration_ms: int | None = None

    # Status
    status: str = "completed"

    # LLM-specific fields
    model: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    cost_usd: float | None = None
    cached_input_tokens: int | None = None
    cache_write_tokens: int | None = None
    # Reasoning/thinking token count — subset of output_tokens.
    # Populated by OpenAI adapter (o-series) and Anthropic extended-thinking.
    reasoning_tokens: int | None = None

    # Payload
    input_data: Any | None = None
    output_data: Any | None = None
    error_message: str | None = None
    # Exception class name (qualified outside builtins) of the error that failed
    # this span. Lets the server group repeated failures by type + message.
    exception_type: str | None = None
    metadata: dict[str, Any] | None = None

    # Context profile (pre-segmented by SDK adapter — Anthropic, LangGraph).
    # When None, the server runs its messages-array classifier on input_data.
    # When [], the SDK explicitly says no segments (server skips classifier).
    # When populated, the server skips classification and persists as-is.
    context_segments: list[dict[str, Any]] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict, dropping None values.

        SpanType enum values are converted to their string representation.
        """
        raw = {
            "span_id": self.span_id,
            "trace_id": self.trace_id,
            "parent_span_id": self.parent_span_id,
            "type": self.type.value if isinstance(self.type, SpanType) else self.type,
            "name": self.name,
            "status": self.status,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_ms": self.duration_ms,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": self.cost_usd,
            "cached_input_tokens": self.cached_input_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "input_data": self.input_data,
            "output_data": self.output_data,
            "error_message": self.error_message,
            "exception_type": self.exception_type,
            "metadata": self.metadata,
            "context_segments": self.context_segments,
        }
        return {k: v for k, v in raw.items() if v is not None}
