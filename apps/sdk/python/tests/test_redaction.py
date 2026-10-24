"""Tests for SDK-side PII redaction (Phase 6f Wave B G-02).

Coverage:
- Per-pattern hit + miss for all eight canonical patterns.
- Mandatory-pattern lockdown (cannot be disabled).
- Custom-pattern injection + validation (cap, length, compile).
- Opt-out via ``redaction_enabled=False``.
- Hook integration: ``Client(redaction_*=...)`` writes redacted strings into
  the transport queue.
- Parity-corpus assertion against the archived G-02 spec corpus —
  the same fixture is consumed by server-side and TS-side tests, so
  byte-equal output is the cross-implementation contract.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

import morse_ai
from morse_ai._internal.redaction import (
    CUSTOM_REPLACEMENT,
    DEFAULT_OFF_PATTERNS,
    MANDATORY_PATTERNS,
    MAX_EXTRA_PATTERN_LEN,
    MAX_EXTRA_PATTERNS,
    PATTERN_KEYS,
    RedactionConfig,
    redact_string,
)
from morse_ai.client import _Client

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def default_config() -> RedactionConfig:
    """Default-on config (all defaults, no extras, no overrides)."""
    return RedactionConfig.build()


@pytest.fixture
def all_patterns_config() -> RedactionConfig:
    """Config with every canonical pattern enabled (including ``ip_address``)."""
    return RedactionConfig(
        enabled=True,
        enabled_patterns=frozenset(PATTERN_KEYS),
        extra_patterns=(),
    )


def _reset() -> None:
    morse_ai._reset_for_testing()


def setup_function() -> None:
    _reset()


def teardown_function() -> None:
    _reset()


# ---------------------------------------------------------------------------
# Pattern hits + misses (per-pattern unit coverage)
# ---------------------------------------------------------------------------


@pytest.mark.req("PHASE-6F-G02")
class TestApiKeyPattern:
    def test_openai_sk_prefix(self, default_config: RedactionConfig) -> None:
        result = redact_string(
            "sk-proj-AbCdEf1234567890ghIjKlMnOpQrStUv", default_config
        )
        assert result == "[REDACTED:api_key]"

    def test_stripe_live(self, default_config: RedactionConfig) -> None:
        result = redact_string("sk_live_51AbCdEfGhIjKlMnOpQrStUvWx", default_config)
        assert result == "[REDACTED:api_key]"

    def test_av_prefix(self, default_config: RedactionConfig) -> None:
        result = redact_string("av_fe9a2ea712345678abcdef0987654321", default_config)
        assert result == "[REDACTED:api_key]"

    def test_aws_access_key(self, default_config: RedactionConfig) -> None:
        result = redact_string("AKIAIOSFODNN7EXAMPLE in env", default_config)
        assert result == "[REDACTED:api_key] in env"

    def test_short_sk_below_threshold_is_not_redacted(
        self, default_config: RedactionConfig
    ) -> None:
        # The prefix exists but the trailing chars don't reach 20.
        result = redact_string("sk-abc nope", default_config)
        assert result == "sk-abc nope"


@pytest.mark.req("PHASE-6F-G02")
class TestJwtPattern:
    def test_three_segment_jwt(self, default_config: RedactionConfig) -> None:
        jwt = (
            "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
            "eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkphbmUgRG9lIn0."
            "xCJVoSi1q19SPSk7kKb6kYP8yU3yz7sLcG5bbqJyU2Q"
        )
        assert redact_string(jwt, default_config) == "[REDACTED:jwt]"

    def test_two_dots_only_is_not_jwt(self, default_config: RedactionConfig) -> None:
        # No eyJ prefix and segments too short.
        assert redact_string("foo.bar.baz", default_config) == "foo.bar.baz"


@pytest.mark.req("PHASE-6F-G02")
class TestCreditCardPattern:
    def test_valid_visa_passes_luhn(self, default_config: RedactionConfig) -> None:
        result = redact_string("4111 1111 1111 1111", default_config)
        assert result == "[REDACTED:credit_card]"

    def test_amex_15_digit(self, default_config: RedactionConfig) -> None:
        result = redact_string("378282246310005", default_config)
        assert result == "[REDACTED:credit_card]"

    def test_random_digits_fail_luhn(self, default_config: RedactionConfig) -> None:
        # Sequential 16 digits don't satisfy Luhn.
        result = redact_string("1234567890123456", default_config)
        assert result == "1234567890123456"


@pytest.mark.req("PHASE-6F-G02")
class TestEmailPattern:
    def test_simple_email(self, default_config: RedactionConfig) -> None:
        assert (
            redact_string("Reach me at jane@example.com tomorrow", default_config)
            == "Reach me at [REDACTED:email] tomorrow"
        )

    def test_email_with_plus_alias(self, default_config: RedactionConfig) -> None:
        assert (
            redact_string("Send to jane+work@sub.example.co.uk", default_config)
            == "Send to [REDACTED:email]"
        )

    def test_no_match_when_not_email(self, default_config: RedactionConfig) -> None:
        assert redact_string("just plain text", default_config) == "just plain text"


@pytest.mark.req("PHASE-6F-G02")
class TestPhonePattern:
    def test_us_dashed(self, default_config: RedactionConfig) -> None:
        assert (
            redact_string("Call 555-867-5309 if needed", default_config)
            == "Call [REDACTED:phone] if needed"
        )

    def test_us_parens(self, default_config: RedactionConfig) -> None:
        assert (
            redact_string("Office line (415) 555-2671", default_config)
            == "Office line [REDACTED:phone]"
        )

    def test_e164(self, default_config: RedactionConfig) -> None:
        assert (
            redact_string("International +44 20 7946 0958 line", default_config)
            == "International [REDACTED:phone] line"
        )


@pytest.mark.req("PHASE-6F-G02")
class TestSsnPattern:
    def test_valid_ssn_dashed(self, default_config: RedactionConfig) -> None:
        assert (
            redact_string("Verify SSN 123-45-6789 on file", default_config)
            == "Verify SSN [REDACTED:ssn] on file"
        )

    def test_invalid_prefix_000(self, default_config: RedactionConfig) -> None:
        assert (
            redact_string("Bad SSN 000-12-3456 should not match", default_config)
            == "Bad SSN 000-12-3456 should not match"
        )

    def test_invalid_prefix_666(self, default_config: RedactionConfig) -> None:
        assert (
            redact_string("Bad SSN 666-12-3456", default_config)
            == "Bad SSN 666-12-3456"
        )


@pytest.mark.req("PHASE-6F-G02")
class TestIpPattern:
    def test_ip_off_by_default(self, default_config: RedactionConfig) -> None:
        assert (
            redact_string("Request from 192.168.1.42", default_config)
            == "Request from 192.168.1.42"
        )

    def test_ip_on_when_explicitly_enabled(
        self, all_patterns_config: RedactionConfig
    ) -> None:
        assert (
            redact_string("Request from 192.168.1.42 yesterday", all_patterns_config)
            == "Request from [REDACTED:ip] yesterday"
        )


# ---------------------------------------------------------------------------
# Multi-pattern + structural cases
# ---------------------------------------------------------------------------


@pytest.mark.req("PHASE-6F-G02")
def test_multiple_patterns_in_one_string(default_config: RedactionConfig) -> None:
    text = "Login as jane@example.com using sk-proj-AbCdEf1234567890ghIjKlMnOpQrStUv"
    assert (
        redact_string(text, default_config)
        == "Login as [REDACTED:email] using [REDACTED:api_key]"
    )


@pytest.mark.req("PHASE-6F-G02")
def test_empty_string_is_passthrough(default_config: RedactionConfig) -> None:
    assert redact_string("", default_config) == ""


@pytest.mark.req("PHASE-6F-G02")
def test_clean_text_is_passthrough(default_config: RedactionConfig) -> None:
    text = "Hello world! Just a plain message about the weather."
    assert redact_string(text, default_config) == text


@pytest.mark.req("PHASE-6F-G02")
def test_redact_value_walks_nested_dict(default_config: RedactionConfig) -> None:
    payload = {
        "input": "use sk-proj-AbCdEf1234567890ghIjKlMnOpQrStUv to call",
        "tokens": 42,
        "ok": True,
        "nested": {"email": "jane@example.com", "count": 1},
        "items": ["clean", "ping jane@example.com again"],
    }
    out = default_config.redact_value(payload)
    assert out == {
        "input": "use [REDACTED:api_key] to call",
        "tokens": 42,
        "ok": True,
        "nested": {"email": "[REDACTED:email]", "count": 1},
        "items": ["clean", "ping [REDACTED:email] again"],
    }


@pytest.mark.req("PHASE-6F-G02")
def test_redact_value_preserves_non_string_leaves(
    default_config: RedactionConfig,
) -> None:
    payload = {"none": None, "bool": False, "float": 3.14, "int": 7}
    assert default_config.redact_value(payload) == payload


@pytest.mark.req("PHASE-6F-G02")
def test_redact_value_depth_bound() -> None:
    # Build a structure deeper than MAX_WALK_DEPTH and confirm we don't recurse
    # forever — the deepest leaves remain untouched (acceptable trade-off vs. a
    # pathological JSON DoS).
    config = RedactionConfig.build()
    leaf = "jane@example.com"
    deep: object = leaf
    for _ in range(20):
        deep = {"x": deep}
    out = config.redact_value(deep)
    # We don't assert exact shape — just that no exception was raised and the
    # top-level dict is still a dict.
    assert isinstance(out, dict)


# ---------------------------------------------------------------------------
# Disabled-patterns + opt-out semantics
# ---------------------------------------------------------------------------


@pytest.mark.req("PHASE-6F-G02")
def test_email_disabled_explicitly() -> None:
    config = RedactionConfig.build(disabled_patterns=["email"])
    assert (
        redact_string("ping jane@example.com today", config)
        == "ping jane@example.com today"
    )
    # Mandatory pattern still runs.
    assert (
        redact_string("creds sk-proj-AbCdEf1234567890ghIjKlMnOpQrStUv", config)
        == "creds [REDACTED:api_key]"
    )


@pytest.mark.req("PHASE-6F-G02")
def test_redaction_disabled_completely() -> None:
    config = RedactionConfig.build(enabled=False)
    text = "creds sk-proj-AbCdEf1234567890ghIjKlMnOpQrStUv and jane@example.com"
    assert redact_string(text, config) == text


@pytest.mark.req("PHASE-6F-G02", "INGEST-ABUSE-09")
@pytest.mark.parametrize("locked", sorted(MANDATORY_PATTERNS))
def test_mandatory_pattern_cannot_be_disabled(locked: str) -> None:
    with pytest.raises(ValueError, match=f"{locked!r}"):
        RedactionConfig.build(disabled_patterns=[locked])


@pytest.mark.req("PHASE-6F-G02")
def test_unknown_pattern_key_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown pattern key"):
        RedactionConfig.build(disabled_patterns=["not_a_real_pattern"])


# ---------------------------------------------------------------------------
# Custom (extra) patterns
# ---------------------------------------------------------------------------


@pytest.mark.req("PHASE-6F-G02")
def test_extra_pattern_replaces_with_custom_label() -> None:
    config = RedactionConfig.build(extra_patterns=[r"INTERNAL-\w+"])
    assert (
        redact_string("INTERNAL-ABC-42 done", config) == f"{CUSTOM_REPLACEMENT}-42 done"
    )


@pytest.mark.req("PHASE-6F-G02")
def test_extra_patterns_capped_at_max() -> None:
    # MAX_EXTRA_PATTERNS=10 — eleven entries must raise.
    too_many = [f"A{i}" for i in range(MAX_EXTRA_PATTERNS + 1)]
    with pytest.raises(ValueError, match="too many"):
        RedactionConfig.build(extra_patterns=too_many)


@pytest.mark.req("PHASE-6F-G02")
def test_extra_pattern_length_capped() -> None:
    too_long = "a" * (MAX_EXTRA_PATTERN_LEN + 1)
    with pytest.raises(ValueError, match="256-char cap"):
        RedactionConfig.build(extra_patterns=[too_long])


@pytest.mark.req("PHASE-6F-G02")
def test_uncompilable_extra_pattern_raises() -> None:
    # Unmatched paren → re.error.
    with pytest.raises(ValueError, match="failed to compile"):
        RedactionConfig.build(extra_patterns=["(unbalanced"])


@pytest.mark.req("PHASE-6F-G02")
def test_extra_pattern_runs_after_builtins() -> None:
    # The user pattern matches a literal that overlaps part of a redacted
    # builtin output; it should NOT match because the builtin replaced the
    # text first.
    config = RedactionConfig.build(extra_patterns=[r"sk-proj"])
    out = redact_string("creds sk-proj-AbCdEf1234567890ghIjKlMnOpQrStUv", config)
    # The api_key builtin wins; no leftover "sk-proj" for the custom pattern.
    assert out == "creds [REDACTED:api_key]"


# ---------------------------------------------------------------------------
# Default-off contract — sanity check on module-level constants
# ---------------------------------------------------------------------------


@pytest.mark.req("PHASE-6F-G02")
def test_default_off_contains_ip_address() -> None:
    assert "ip_address" in DEFAULT_OFF_PATTERNS


@pytest.mark.req("PHASE-6F-G02")
def test_default_config_excludes_ip_address() -> None:
    config = RedactionConfig.build()
    assert "ip_address" not in config.enabled_patterns
    assert "api_key_prefixed" in config.enabled_patterns


# ---------------------------------------------------------------------------
# Hook-point integration via _Client / transport
# ---------------------------------------------------------------------------


@pytest.mark.req("PHASE-6F-G02")
def test_client_redacts_payload_before_queue() -> None:
    """A Client constructed with defaults rewrites string leaves before they
    reach the transport queue."""
    client = _Client(
        api_key="av_test_redact",
        endpoint="http://localhost:9999/api/v1/traces/batch",
    )
    try:
        # We don't want the background thread to drain it; .shutdown() will
        # flush — but inspection happens BEFORE shutdown by peeking the queue
        # directly. The queue.put returns immediately so we read what we put.
        client.transport.track(
            {
                "agent": "ops",
                "input": "use sk-proj-AbCdEf1234567890ghIjKlMnOpQrStUv now",
                "metadata": {"contact": "jane@example.com", "ok": True},
            }
        )
        # Drain just one event from the queue WITHOUT going through the HTTP
        # client — which is unreachable on port 9999.
        drained = client.transport._queue.drain(1)
        assert len(drained) == 1
        event = drained[0]
        assert event["input"] == "use [REDACTED:api_key] now"
        assert event["metadata"]["contact"] == "[REDACTED:email]"
        assert event["metadata"]["ok"] is True
    finally:
        client.shutdown()


@pytest.mark.req("PHASE-6F-G02")
def test_client_with_redaction_disabled_passes_through_at_sdk() -> None:
    client = _Client(
        api_key="av_test_optout",
        endpoint="http://localhost:9999/api/v1/traces/batch",
        redaction_enabled=False,
    )
    try:
        sensitive = "sk-proj-AbCdEf1234567890ghIjKlMnOpQrStUv"
        client.transport.track({"input": sensitive})
        drained = client.transport._queue.drain(1)
        assert drained[0]["input"] == sensitive
    finally:
        client.shutdown()


@pytest.mark.req("PHASE-6F-G02")
def test_client_construction_rejects_disabling_mandatory_pattern() -> None:
    with pytest.raises(ValueError, match="cannot be disabled"):
        _Client(
            api_key="av_test_lock",
            endpoint="http://localhost:9999/api/v1/traces/batch",
            redaction_disabled_patterns=["jwt"],
        )


@pytest.mark.req("PHASE-6F-G02")
def test_client_construction_rejects_uncompilable_extra_pattern() -> None:
    with pytest.raises(ValueError, match="failed to compile"):
        _Client(
            api_key="av_test_bad_regex",
            endpoint="http://localhost:9999/api/v1/traces/batch",
            redaction_extra_patterns=["(unbalanced"],
        )


@pytest.mark.req("PHASE-6F-G02")
def test_init_rejects_disabling_mandatory_pattern() -> None:
    """The public ``init()`` surface MUST raise on misconfig — not silently
    swallow it through the absorb_errors decorator."""
    with pytest.raises(ValueError, match="cannot be disabled"):
        morse_ai.init(
            api_key="av_init_bad",
            endpoint="http://localhost:9999/api/v1/traces/batch",
            redaction_disabled_patterns=["api_key_prefixed"],
        )


@pytest.mark.req("PHASE-6F-G02")
def test_init_rejects_uncompilable_extra_pattern() -> None:
    with pytest.raises(ValueError, match="failed to compile"):
        morse_ai.init(
            api_key="av_init_bad_regex",
            endpoint="http://localhost:9999/api/v1/traces/batch",
            redaction_extra_patterns=["(unbalanced"],
        )


# ---------------------------------------------------------------------------
# Parity corpus — the cross-implementation contract
# ---------------------------------------------------------------------------


# Keep the shared parity fixture inside the independent SDK repository.
_CORPUS_PATH = Path(__file__).parent / "fixtures" / "parity_corpus.json"


def _load_corpus() -> list[dict[str, object]]:
    with _CORPUS_PATH.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    return data["rows"]


@pytest.mark.req("PHASE-6F-G02")
@pytest.mark.parametrize("row", _load_corpus(), ids=lambda r: str(r["id"]))
def test_parity_corpus_matches_canonical_output(row: dict[str, object]) -> None:
    """Every row in the parity corpus must produce the spec's expected output.

    The same fixture is consumed by the server-side redactor + the TS SDK; if
    this test fails for a row, one of the three implementations has drifted
    from the canonical pattern catalogue.
    """
    enabled = list(row["enabled_patterns"])  # type: ignore[arg-type]
    if enabled:
        config = RedactionConfig(
            enabled=True,
            enabled_patterns=frozenset(enabled),
            extra_patterns=(),
        )
    else:
        # Empty list → "all defaults" per the corpus README.
        config = RedactionConfig.build()

    actual = redact_string(row["input"], config)  # type: ignore[arg-type]
    assert actual == row["expected_output"], (
        f"Row {row['id']!r}: expected {row['expected_output']!r}, got {actual!r}"
    )


# ---------------------------------------------------------------------------
# Stable iteration order
# ---------------------------------------------------------------------------


@pytest.mark.req("PHASE-6F-G02")
def test_pattern_keys_are_canonical_order() -> None:
    """The iteration order documented in design.md §2 footer must be stable."""
    assert PATTERN_KEYS == (
        "api_key_prefixed",
        "jwt",
        "credit_card",
        "email",
        "phone_us",
        "phone_e164",
        "ssn_us",
        "ip_address",
        "secret_assignment",
    )


@pytest.mark.req("PHASE-6F-G02")
def test_all_patterns_are_compilable_python_regex() -> None:
    """Self-check: every pattern source compiles as a Python regex (would-be
    smoke test if a future edit introduces a typo)."""
    from morse_ai._internal.redaction import _COMPILED  # noqa: SLF001

    for key in PATTERN_KEYS:
        assert isinstance(_COMPILED[key], re.Pattern)


# ---------------------------------------------------------------------------
# MHQ-1412 — identifiers are not PII (dogfood 2026-10-18, groups A/D/F)
# ---------------------------------------------------------------------------

_ID_CASES = [
    # FAISS document id (group A, trace f18d58b5): "...-0c[REDACTED:phone]"
    '{"id": "bc41d39b-d298-4de5-83ba-0c4576313225"}',
    # duplicate_of_span_id (group D): leading digits were Luhn-valid
    '{"duplicate_of_span_id": "40128888-8888-1881-286c-7f355e8c7903"}',
    # 32-hex trace id (group F): "...7f[REDACTED:phone]"
    "trace_id=50df6d43791e4443b23f7f4576313225",
    # 16-hex span id with a 10-digit run
    "span 4c4155551234abcd",
]


@pytest.mark.req("MHQ-1412")
@pytest.mark.parametrize("text", _ID_CASES)
def test_ids_are_not_redacted_as_phone_or_card(text: str) -> None:
    config = RedactionConfig.build(enabled=True)
    assert redact_string(text, config) == text


@pytest.mark.req("MHQ-1412")
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "call 415-555-1234 re 2d16e332-1935-4a18-8568-ddf9431a1c3c",
            "call [REDACTED:phone] re 2d16e332-1935-4a18-8568-ddf9431a1c3c",
        ),
        ("card 4111111111111111", "card [REDACTED:credit_card]"),
        ("ssn 123-45-6789", "ssn [REDACTED:ssn]"),
        ("deadbeefdeadbeef12@example.com", "[REDACTED:email]"),
        # A hex key body must never be shielded from the key patterns.
        (
            "ghp_1234567890abcdef1234567890abcdef1234",  # pragma: allowlist secret
            "[REDACTED:api_key]",
        ),
    ],
)
def test_pii_next_to_ids_is_still_redacted(text: str, expected: str) -> None:
    config = RedactionConfig.build(enabled=True)
    assert redact_string(text, config) == expected
