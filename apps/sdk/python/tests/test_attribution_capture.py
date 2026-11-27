"""attribution-capture (MHQ-1275) — SDK side of ATTR-01, ATTR-04, ATTR-06.

Record keys (``link_record`` / ``run(record_keys=)``), declared agent version
(``init`` / ``run``), prompt references (``set_prompt``), git revision
precedence and ``service.version`` from ``OTEL_RESOURCE_ATTRIBUTES``, on both
the JSON payload and the OTLP spans.
"""

from __future__ import annotations

import logging

import pytest

import amund
from amund import tracing
from amund._internal import attribution, env
from amund._internal.redaction import RedactionConfig
from amund.otel_provider import get_engine

_KEY = "av_test_key"  # pragma: allowlist secret
# Never the default production endpoint: a fake key there returns 401, and the
# abandoned background exporter records an auth_rejected drop that leaks into
# the global diagnostics counters other tests drain.
_DEAD_ENDPOINT = "http://localhost:0/api/v1/traces/batch"

_GIT_VARS = (
    "MORSE_GIT_REVISION",
    "GITHUB_SHA",
    "CI_COMMIT_SHA",
    "RAILWAY_GIT_COMMIT_SHA",
    "VERCEL_GIT_COMMIT_SHA",
    "RENDER_GIT_COMMIT",
    "SOURCE_VERSION",
)


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for var in (*_GIT_VARS, "OTEL_RESOURCE_ATTRIBUTES", "OTEL_SERVICE_NAME"):
        monkeypatch.delenv(var, raising=False)
    amund._reset_for_testing()
    tracing._transport = None
    tracing._last_trace = None
    yield
    amund._reset_for_testing()


@pytest.fixture
def engine():
    """OTel engine with an in-memory recorder (env must be set BEFORE use)."""
    eng = get_engine()
    eng.configure(
        endpoint="http://127.0.0.1:1/api/v1/otlp/v1/traces",
        api_key="av_test",  # pragma: allowlist secret
        sample_rate=1.0,
        schedule_delay_millis=50,
        max_queue_size=4096,
        max_export_batch_size=512,
        max_field_bytes=65536,
        timeout=1.0,
        redaction_config=RedactionConfig.build(enabled=True),
        capture_for_tests=True,
    )
    yield eng
    eng.reset_for_testing()


def _otel_spans(eng, trace_id_hex):
    eng.provider.force_flush()
    tid = int(trace_id_hex, 16)
    spans = [s for s in eng.recorder.spans if s.context.trace_id == tid]
    assert spans
    return spans


def _otel_root(eng, trace_id_hex):
    spans = _otel_spans(eng, trace_id_hex)
    return next(s for s in spans if "amund.agent" in dict(s.attributes))


def _trace() -> dict:
    payload = tracing._get_last_trace()
    assert payload is not None
    return payload["trace"]


# ---------------------------------------------------------------------------
# ATTR-01 — record keys
# ---------------------------------------------------------------------------


@pytest.mark.req("ATTR-01")
def test_link_record_exported_from_package():
    assert callable(amund.link_record)
    assert "link_record" in amund.__all__


@pytest.mark.req("ATTR-01")
def test_link_record_outside_trace_is_noop_with_debug_log(caplog):
    with caplog.at_level(logging.DEBUG, logger="amund"):
        amund.link_record("zendesk_ticket", "4812")  # must not raise
    assert "no open trace" in caplog.text


@pytest.mark.req("ATTR-01")
def test_link_record_accumulates_and_lands_on_json_trace():
    with amund.run("agent"):
        amund.link_record("zendesk_ticket", "4812")
        amund.link_record("order_id", 99)
    assert _trace()["record_keys"] == {"zendesk_ticket": "4812", "order_id": "99"}


@pytest.mark.req("ATTR-01")
def test_link_record_repeated_type_replaces_value():
    with amund.run("agent"):
        amund.link_record("ticket", "1")
        amund.link_record("ticket", "2")
    assert _trace()["record_keys"] == {"ticket": "2"}


@pytest.mark.req("ATTR-01")
def test_link_record_works_under_trace_agent():
    @amund.trace_agent("decorated")
    def go():
        amund.link_record("ticket", "7")

    go()
    assert _trace()["record_keys"] == {"ticket": "7"}


@pytest.mark.req("ATTR-01")
def test_run_record_keys_kwarg_merges_with_link_record():
    with amund.run("agent", record_keys={"Ticket": 12, "crm.deal": "d-1"}):
        amund.link_record("order_id", "o-9")
    assert _trace()["record_keys"] == {
        "ticket": "12",
        "crm.deal": "d-1",
        "order_id": "o-9",
    }


@pytest.mark.req("ATTR-01")
def test_key_type_is_lowercased_and_value_trimmed():
    with amund.run("agent"):
        amund.link_record("  Zendesk_Ticket ", "  4812  ")
    assert _trace()["record_keys"] == {"zendesk_ticket": "4812"}


@pytest.mark.req("ATTR-01")
@pytest.mark.parametrize(
    ("key_type", "key_value"),
    [
        ("", "x"),
        ("-leading-dash", "x"),
        ("has space", "x"),
        ("UPPER!", "x"),
        ("a" * 65, "x"),
        (None, "x"),
        (5, "x"),
        ("ok", ""),
        ("ok", "   "),
        ("ok", None),
        ("ok", True),
        ("ok", "v" * 257),
    ],
)
def test_invalid_keys_are_dropped_without_raising(key_type, key_value, caplog):
    with caplog.at_level(logging.DEBUG, logger="amund"), amund.run("agent"):
        amund.link_record(key_type, key_value)  # type: ignore[arg-type]
    assert "record_keys" not in _trace()
    assert "dropped" in caplog.text


@pytest.mark.req("ATTR-01")
def test_boundary_lengths_are_accepted():
    with amund.run("agent"):
        amund.link_record("a" * 64, "v" * 256)
    assert _trace()["record_keys"] == {"a" * 64: "v" * 256}


@pytest.mark.req("ATTR-01")
def test_key_cap_of_20_per_trace():
    with amund.run("agent"):
        for i in range(25):
            amund.link_record(f"k{i}", str(i))
        # an existing type can still be updated once the cap is reached
        amund.link_record("k0", "updated")
    keys = _trace()["record_keys"]
    assert len(keys) == attribution.MAX_RECORD_KEYS == 20
    assert keys["k0"] == "updated"
    assert "k20" not in keys


@pytest.mark.req("ATTR-01")
def test_hostile_value_str_never_escapes():
    class Boom:
        def __str__(self) -> str:
            raise RuntimeError("customer __str__ exploded")

    with amund.run("agent", record_keys={"bad": Boom(), "good": "1"}):  # type: ignore[dict-item]
        amund.link_record("also_bad", Boom())  # type: ignore[arg-type]
    assert _trace()["record_keys"] == {"good": "1"}


@pytest.mark.req("ATTR-01")
def test_malformed_record_keys_kwarg_never_fails_trace_start():
    with amund.run("agent", record_keys=["not", "a", "dict"]):  # type: ignore[arg-type]
        pass
    assert "record_keys" not in _trace()


@pytest.mark.req("ATTR-01")
def test_no_attribution_fields_when_nothing_declared():
    with amund.run("agent"):
        pass
    trace = _trace()
    for field in ("record_keys", "agent_version", "prompts"):
        assert field not in trace


@pytest.mark.req("ATTR-01")
def test_record_keys_become_root_span_attributes_on_otlp(engine):
    with amund.run("agent", record_keys={"order_id": 5}) as r:
        amund.link_record("zendesk_ticket", "4812")
        trace_id = r.trace_id
    attrs = dict(_otel_root(engine, trace_id).attributes)
    assert attrs["amund.record_key.order_id"] == "5"
    assert attrs["amund.record_key.zendesk_ticket"] == "4812"


# ---------------------------------------------------------------------------
# ATTR-04 — agent_version
# ---------------------------------------------------------------------------


@pytest.mark.req("ATTR-04")
def test_init_agent_version_is_default_for_every_trace():
    amund.init(api_key=_KEY, endpoint=_DEAD_ENDPOINT, agent_version="2.4.1")
    with amund.run("agent"):
        pass
    assert _trace()["agent_version"] == "2.4.1"


@pytest.mark.req("ATTR-04")
def test_run_agent_version_overrides_init_default():
    amund.init(api_key=_KEY, endpoint=_DEAD_ENDPOINT, agent_version="2.4.1")
    with amund.run("agent", agent_version="2.5.0-rc1"):
        pass
    assert _trace()["agent_version"] == "2.5.0-rc1"
    with amund.run("agent"):
        pass
    assert _trace()["agent_version"] == "2.4.1"


@pytest.mark.req("ATTR-04")
def test_agent_version_applies_to_trace_agent_decorator():
    amund.init(api_key=_KEY, endpoint=_DEAD_ENDPOINT, agent_version="9.9")

    @amund.trace_agent("decorated")
    def go():
        return 1

    go()
    assert _trace()["agent_version"] == "9.9"


@pytest.mark.req("ATTR-04")
def test_agent_version_is_stripped_truncated_and_blank_ignored():
    with amund.run("agent", agent_version="  v" + "1" * 200 + "  "):
        pass
    value = _trace()["agent_version"]
    assert len(value) == attribution.MAX_AGENT_VERSION_LEN == 128
    assert value.startswith("v1")
    with amund.run("agent", agent_version="   "):
        pass
    assert "agent_version" not in _trace()


@pytest.mark.req("ATTR-04")
def test_agent_version_is_root_span_attribute_on_otlp(engine):
    with amund.run("agent", agent_version="2.4.1") as r:
        with amund.span("child"):
            pass
        trace_id = r.trace_id
    spans = _otel_spans(engine, trace_id)
    versions = [dict(s.attributes).get("amund.agent_version") for s in spans]
    assert versions.count("2.4.1") == 1  # root only
    root = next(s for s in spans if dict(s.attributes).get("amund.agent"))
    assert dict(root.attributes)["amund.agent_version"] == "2.4.1"


# ---------------------------------------------------------------------------
# ATTR-04 — set_prompt
# ---------------------------------------------------------------------------


@pytest.mark.req("ATTR-04")
def test_set_prompt_exported_from_package():
    assert callable(amund.set_prompt)
    assert "set_prompt" in amund.__all__


@pytest.mark.req("ATTR-04")
def test_set_prompt_outside_trace_is_noop_with_debug_log(caplog):
    with caplog.at_level(logging.DEBUG, logger="amund"):
        amund.set_prompt("triage", 7)
    assert "no open trace" in caplog.text


@pytest.mark.req("ATTR-04")
def test_set_prompt_with_and_without_version_at_trace_level():
    with amund.run("agent"):
        amund.set_prompt("triage", version=7)
        amund.set_prompt("summarise")
    assert _trace()["prompts"] == [
        {"name": "triage", "version": 7},
        {"name": "summarise"},
    ]


@pytest.mark.req("ATTR-04")
def test_set_prompt_inside_span_pins_the_span_id():
    with amund.run("agent"), amund.span("llm-call"):
        amund.set_prompt("triage", 3)
    payload = tracing._get_last_trace()
    assert payload is not None
    child = next(s for s in payload["spans"] if s["name"] == "llm-call")
    assert payload["trace"]["prompts"] == [
        {"name": "triage", "version": 3, "span_id": child["span_id"]}
    ]


@pytest.mark.req("ATTR-04")
def test_set_prompt_after_span_closes_is_trace_level_again():
    with amund.run("agent"):
        with amund.span("llm-call"):
            pass
        amund.set_prompt("triage")
    assert "span_id" not in _trace()["prompts"][0]


@pytest.mark.req("ATTR-04")
def test_set_prompt_duplicates_collapse():
    with amund.run("agent"):
        amund.set_prompt("triage", 1)
        amund.set_prompt("triage", 1)
        amund.set_prompt("triage", 2)
    assert _trace()["prompts"] == [
        {"name": "triage", "version": 1},
        {"name": "triage", "version": 2},
    ]


@pytest.mark.req("ATTR-04")
@pytest.mark.parametrize("bad_version", [0, -1, "latest", "2026-10-01", 1.5, True])
def test_invalid_version_is_dropped_but_name_kept(bad_version):
    with amund.run("agent"):
        amund.set_prompt("triage", bad_version)  # type: ignore[arg-type]
    assert _trace()["prompts"] == [{"name": "triage"}]


@pytest.mark.req("ATTR-04", "MHQ-1415")
@pytest.mark.parametrize(("given", "expected"), [("7", 7), ("v12", 12), (" 3 ", 3)])
def test_whole_number_string_version_is_read_not_dropped(given, expected):
    """MHQ-1415: a string version used to vanish silently."""
    with amund.run("agent"):
        amund.set_prompt("triage", given)
    assert _trace()["prompts"] == [{"name": "triage", "version": expected}]


@pytest.mark.req("MHQ-1415")
def test_unreadable_version_warns_once(caplog, monkeypatch):
    from amund._internal import attribution

    monkeypatch.setattr(attribution, "_prompt_version_warned", False)
    with caplog.at_level("WARNING", logger="amund"), amund.run("agent"):
        amund.set_prompt("a", "latest")
        amund.set_prompt("b", "latest")
    warnings = [r for r in caplog.records if "set_prompt() version" in r.getMessage()]
    assert len(warnings) == 1


@pytest.mark.req("ATTR-04")
@pytest.mark.parametrize("bad_name", ["", "   ", None, 5, "n" * 256])
def test_invalid_name_is_dropped_without_raising(bad_name):
    with amund.run("agent"):
        amund.set_prompt(bad_name, 1)  # type: ignore[arg-type]
    assert "prompts" not in _trace()


@pytest.mark.req("ATTR-04")
def test_prompt_ref_cap_of_20_per_trace():
    with amund.run("agent"):
        for i in range(30):
            amund.set_prompt(f"p{i}", 1)
    prompts = _trace()["prompts"]
    assert len(prompts) == attribution.MAX_PROMPT_REFS == 20
    assert prompts[-1]["name"] == "p19"


@pytest.mark.req("ATTR-04")
def test_otlp_prompt_on_root_span_when_set_outside_a_span(engine):
    with amund.run("agent") as r:
        amund.set_prompt("triage", 7)
        trace_id = r.trace_id
    attrs = dict(_otel_root(engine, trace_id).attributes)
    assert attrs["amund.prompt.name"] == "triage"
    assert attrs["amund.prompt.version"] == 7


@pytest.mark.req("ATTR-04")
def test_otlp_prompt_without_version_has_no_version_attr(engine):
    with amund.run("agent") as r:
        amund.set_prompt("triage")
        trace_id = r.trace_id
    attrs = dict(_otel_root(engine, trace_id).attributes)
    assert attrs["amund.prompt.name"] == "triage"
    assert "amund.prompt.version" not in attrs


@pytest.mark.req("ATTR-04")
def test_otlp_prompt_lands_on_the_span_it_was_set_in(engine):
    with amund.run("agent") as r:
        amund.set_prompt("root-prompt", 1)
        with amund.span("llm-call"):
            amund.set_prompt("span-prompt", 2)
        trace_id = r.trace_id
    spans = {s.name: dict(s.attributes) for s in _otel_spans(engine, trace_id)}
    assert spans["agent"]["amund.prompt.name"] == "root-prompt"
    assert spans["agent"]["amund.prompt.version"] == 1
    assert spans["llm-call"]["amund.prompt.name"] == "span-prompt"
    assert spans["llm-call"]["amund.prompt.version"] == 2


@pytest.mark.req("ATTR-04")
def test_otlp_last_prompt_per_span_wins_but_json_keeps_all(engine):
    with amund.run("agent") as r:
        with amund.span("llm-call"):
            amund.set_prompt("first", 1)
            amund.set_prompt("second", 2)
        trace_id = r.trace_id
    spans = {s.name: dict(s.attributes) for s in _otel_spans(engine, trace_id)}
    assert spans["llm-call"]["amund.prompt.name"] == "second"
    assert "amund.prompt.name" not in spans["agent"]  # no overwrite elsewhere
    assert [p["name"] for p in _trace()["prompts"]] == ["first", "second"]


@pytest.mark.req("ATTR-04")
def test_otlp_version_does_not_leak_from_a_versionless_later_ref(engine):
    with amund.run("agent") as r:
        amund.set_prompt("a", 5)
        amund.set_prompt("b")
        trace_id = r.trace_id
    attrs = dict(_otel_root(engine, trace_id).attributes)
    assert attrs["amund.prompt.name"] == "b"
    assert "amund.prompt.version" not in attrs


# ---------------------------------------------------------------------------
# ATTR-06 — git revision precedence
# ---------------------------------------------------------------------------


@pytest.mark.req("ATTR-06")
def test_git_revision_precedence_order(monkeypatch):
    # Set from lowest to highest priority; each step must outrank all before.
    for var in reversed(_GIT_VARS):
        monkeypatch.setenv(var, f"sha-{var}")
        assert env.get_git_revision() == f"sha-{var}"


@pytest.mark.req("ATTR-06")
def test_morse_git_revision_alone_needs_no_git_binary(monkeypatch):
    monkeypatch.setenv("MORSE_GIT_REVISION", "abc123")
    monkeypatch.setenv("PATH", "")  # no git binary reachable
    assert env.get_git_revision() == "abc123"


@pytest.mark.req("ATTR-06")
def test_source_version_is_used_before_git_subprocess(monkeypatch):
    monkeypatch.setenv("SOURCE_VERSION", "heroku-sha")
    monkeypatch.setenv("PATH", "")
    assert env.get_git_revision() == "heroku-sha"


@pytest.mark.req("ATTR-06")
def test_blank_env_values_are_skipped(monkeypatch):
    monkeypatch.setenv("MORSE_GIT_REVISION", "   ")
    monkeypatch.setenv("GITHUB_SHA", "real")
    assert env.get_git_revision() == "real"


@pytest.mark.req("ATTR-06")
def test_git_revision_never_raises_without_git(monkeypatch):
    monkeypatch.setenv("PATH", "")
    assert env.get_git_revision() is None


@pytest.mark.req("ATTR-06")
def test_morse_git_revision_reaches_json_resource_and_otlp_resource(
    monkeypatch, engine
):
    monkeypatch.setenv("MORSE_GIT_REVISION", "abc123")
    amund.init(api_key=_KEY, endpoint=_DEAD_ENDPOINT)
    with amund.run("agent"):
        pass
    payload = tracing._get_last_trace()
    assert payload is not None
    assert payload["resource"]["git.revision"] == "abc123"

    # The OTLP resource is built at configure() time, after the env is set.
    engine.configure(
        endpoint="http://127.0.0.1:1/api/v1/otlp/v1/traces",
        api_key="av_test",  # pragma: allowlist secret
        sample_rate=1.0,
        schedule_delay_millis=50,
        max_queue_size=4096,
        max_export_batch_size=512,
        max_field_bytes=65536,
        timeout=1.0,
        redaction_config=RedactionConfig.build(enabled=True),
        capture_for_tests=True,
    )
    with amund.run("agent") as r:
        trace_id = r.trace_id
    resource = dict(_otel_root(engine, trace_id).resource.attributes)
    assert resource["git.revision"] == "abc123"


# ---------------------------------------------------------------------------
# ATTR-06 — service.version from OTEL_RESOURCE_ATTRIBUTES
# ---------------------------------------------------------------------------


@pytest.mark.req("ATTR-06")
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("service.version=2.4.1", "2.4.1"),
        ("deployment.environment=prod,service.version=2.4.1,team=a", "2.4.1"),
        (" service.version = 1.0.0 ", "1.0.0"),
        ("service.version=1.0.0%2Bbuild.7", "1.0.0+build.7"),
        ("service.version=a%20b", "a b"),
        ("service.name=x", None),
        ("service.version=", None),
        ("service.version", None),
        ("", None),
        ("garbage,,=,", None),
    ],
)
def test_service_version_parsing(monkeypatch, raw, expected):
    monkeypatch.setenv("OTEL_RESOURCE_ATTRIBUTES", raw)
    assert env.get_service_version() == expected


@pytest.mark.req("ATTR-06")
def test_service_version_unset_is_none():
    assert env.get_service_version() is None


@pytest.mark.req("ATTR-06")
def test_service_version_is_capped_at_128(monkeypatch):
    monkeypatch.setenv("OTEL_RESOURCE_ATTRIBUTES", "service.version=" + "9" * 300)
    assert env.get_service_version() == "9" * 128


@pytest.mark.req("ATTR-06")
def test_service_version_reaches_json_resource(monkeypatch):
    monkeypatch.setenv("OTEL_RESOURCE_ATTRIBUTES", "service.version=2.4.1")
    amund.init(api_key=_KEY, endpoint=_DEAD_ENDPOINT)
    with amund.run("agent"):
        pass
    payload = tracing._get_last_trace()
    assert payload is not None
    assert payload["resource"]["service.version"] == "2.4.1"


@pytest.mark.req("ATTR-06")
def test_service_version_absent_from_json_resource_when_unset(monkeypatch):
    monkeypatch.setenv("MORSE_GIT_REVISION", "abc")  # force a resource dict
    amund.init(api_key=_KEY, endpoint=_DEAD_ENDPOINT)
    with amund.run("agent"):
        pass
    payload = tracing._get_last_trace()
    assert payload is not None
    assert "service.version" not in payload["resource"]


@pytest.mark.req("ATTR-06")
def test_service_version_reaches_otlp_resource(monkeypatch, engine):
    monkeypatch.setenv("OTEL_RESOURCE_ATTRIBUTES", "service.version=2.4.1%2Bx")
    engine.configure(
        endpoint="http://127.0.0.1:1/api/v1/otlp/v1/traces",
        api_key="av_test",  # pragma: allowlist secret
        sample_rate=1.0,
        schedule_delay_millis=50,
        max_queue_size=4096,
        max_export_batch_size=512,
        max_field_bytes=65536,
        timeout=1.0,
        redaction_config=RedactionConfig.build(enabled=True),
        capture_for_tests=True,
    )
    with amund.run("agent") as r:
        trace_id = r.trace_id
    resource = dict(_otel_root(engine, trace_id).resource.attributes)
    assert resource["service.version"] == "2.4.1+x"


@pytest.mark.req("ATTR-06")
def test_service_name_in_env_still_never_becomes_agent_identity(monkeypatch, engine):
    """Reading OTEL_RESOURCE_ATTRIBUTES for service.version must not change MHQ-1100."""
    monkeypatch.setenv(
        "OTEL_RESOURCE_ATTRIBUTES", "service.version=1,service.name=proc"
    )
    with amund.run("explicit") as r:
        trace_id = r.trace_id
    attrs = dict(_otel_root(engine, trace_id).attributes)
    assert attrs["amund.agent"] == "explicit"


# ---------------------------------------------------------------------------
# Non-invasiveness
# ---------------------------------------------------------------------------


@pytest.mark.req("ATTR-01", "ATTR-04")
def test_nothing_raises_on_garbage_input():
    with amund.run("agent", record_keys={1: 2, None: None}, agent_version=object()):  # type: ignore[dict-item,arg-type]
        amund.link_record(object(), object())  # type: ignore[arg-type]
        amund.set_prompt(object(), object())  # type: ignore[arg-type]
        amund.set_prompt("ok", version=[1])  # type: ignore[arg-type]
    assert _trace()["prompts"] == [{"name": "ok"}]
