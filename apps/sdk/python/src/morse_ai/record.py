"""RunRecord — the core data unit for Morse telemetry."""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

from morse_ai._version import __version__


@dataclass
class RunRecord:
    """A single execution of an agent with outcome data.

    This is the data unit sent to the Morse backend via
    POST /api/v1/runs/batch. The SDK creates these internally
    from morse_ai.record() calls.
    """

    agent: str
    status: str = "completed"
    success: bool | None = None
    outcome: str | None = None
    model: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    cost_usd: float | None = None
    duration_ms: int | None = None
    error_message: str | None = None
    metadata: dict[str, Any] | None = None
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    started_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    ended_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    sdk_version: str = __version__

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict, dropping None values."""
        d = asdict(self)
        return {k: v for k, v in d.items() if v is not None}
