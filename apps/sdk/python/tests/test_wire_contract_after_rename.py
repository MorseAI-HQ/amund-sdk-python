"""Tripwire: the Morse -> Amund package rename must not change the wire contract.

The import package moved from ``morse_ai`` to ``amund``, but the server keys on
these strings, so they stay exactly as they were. If one of these assertions
fails, the change needs a coordinated API release, not an SDK-only edit.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from amund import client, otel_provider

_SRC = Path(__file__).resolve().parent.parent / "src" / "amund"


def _source(rel: str) -> str:
    return (_SRC / rel).read_text()


def test_otel_instrumentation_scope_name_unchanged():
    assert otel_provider._INSTRUMENTATION_NAME == "morse_ai"


def test_default_endpoint_host_unchanged():
    assert client._DEFAULT_ENDPOINT == "https://api.morseai.dev/api/v1/traces/batch"


@pytest.mark.parametrize(
    ("rel", "literal"),
    [
        # Resource attributes the server reads.
        ("tracing.py", 'resource["morse_ai.source"]'),
        ("tracing.py", 'resource["morse_ai.framework"]'),
        ("otel_provider.py", 'resource_dict.get("morse_ai.framework")'),
        ("otel_provider.py", 'resource_dict.get("morse_ai.source")'),
        # HTTP headers.
        ("transport.py", '"User-Agent": f"morse_ai-python/{__version__}"'),
        ("_internal/diagnostics.py", '"User-Agent": f"morse_ai-python/{__version__}"'),
        ("_internal/diagnostics.py", '"X-Morse-Diagnostics"'),
        ("transport.py", '"X-Morse-Quota-Limit"'),
        # Default service name on log records.
        ("logging.py", '"morse_ai.sdk"'),
        ("client.py", '"morse_ai.sdk"'),
        # Diagnostics fingerprints the server groups on.
        ("__init__.py", 'fingerprint="morse_ai.init"'),
        ("tracing.py", '"morse_ai.tracing.run_exit"'),
        ("_internal/diagnostics.py", '"morse_ai.transport.delivery"'),
        ("adapters/openai.py", 'fingerprint="morse_ai.adapters.openai.install"'),
    ],
)
def test_wire_literal_unchanged(rel, literal):
    assert literal in _source(rel), f"{literal} missing from src/amund/{rel}"
