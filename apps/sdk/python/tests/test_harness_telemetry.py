"""Harness telemetry v1 (H2-C5) — optional, never-raising, schema-exact.

Contract: ``tests/fixtures/harness2/harness-telemetry-v1.schema.json`` (vendored
verbatim, frozen). Covers omission, valid values, every invalid case, the OTLP
keys/types, schema validation of the emitted payload and the pinned
``file_set_hash`` vectors.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import unicodedata
from pathlib import Path

import pytest

import amund
from amund import tracing
from amund._internal import harness
from amund._internal.redaction import RedactionConfig
from amund.otel_provider import get_engine

pytestmark = pytest.mark.req("H2-C5")

_KEY = "av_test_key"  # pragma: allowlist secret  # noqa: E501
_DEAD_ENDPOINT = "http://localhost:0/api/v1/traces/batch"
SCHEMA_PATH = (
    Path(__file__).parent / "fixtures/harness2/harness-telemetry-v1.schema.json"
)
A = "a" * 64
B = "b" * 64
C = "c" * 64

# Pinned cross-language vectors (amund-core schemas/test_harness2_contracts.py).
FILE_SET_VECTOR_FILES = {
    "prompts/system.md": b"You are a support agent.\n",
    "skills/refund/SKILL.md": b"# Refund\r\nSteps\n",  # CRLF kept: raw bytes
    "scripts/check.py": b"",
}
FILE_SET_VECTOR = "6cfb940903f0610c8f4107a0051496e6a3066cc43e511718cfeadbd007da9457"  # pragma: allowlist secret  # noqa: E501
EMPTY_SET_VECTOR = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"  # pragma: allowlist secret  # noqa: E501

FIELDS = (
    "harness_hash",
    "harness_version",
    "harness_channels_changed",
    "harness_channels_incomplete",
    "harness_parent_hash",
    "harness_run_role",
)


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for var in ("OTEL_RESOURCE_ATTRIBUTES", "OTEL_SERVICE_NAME"):
        monkeypatch.delenv(var, raising=False)
    amund._reset_for_testing()
    tracing._transport = None
    tracing._last_trace = None
    yield
    amund._reset_for_testing()


@pytest.fixture
def engine():
    eng = get_engine()
    eng.configure(
        endpoint="http://127.0.0.1:1/api/v1/otlp/v1/traces",
        api_key="av_test",  # pragma: allowlist secret  # noqa: E501
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


def _trace() -> dict:
    payload = tracing._get_last_trace()
    assert payload is not None
    return payload["trace"]


def _harness_of(trace: dict) -> dict:
    return {k: v for k, v in trace.items() if k.startswith("harness_")}


def _root_attrs(eng, trace_id_hex: str) -> dict:
    eng.provider.force_flush()
    tid = int(trace_id_hex, 16)
    spans = [s for s in eng.recorder.spans if s.context.trace_id == tid]
    root = next(s for s in spans if "amund.agent" in dict(s.attributes))
    return dict(root.attributes)


def _validator():
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(SCHEMA_PATH.read_text())
    return jsonschema.Draft202012Validator(schema)


# ---------------------------------------------------------------------------
# Omission: nothing emitted, nothing stored
# ---------------------------------------------------------------------------


def test_omitted_fields_add_nothing_to_json_or_otlp(engine):
    with amund.run("agent") as r:
        pass
    assert _harness_of(_trace()) == {}
    attrs = _root_attrs(engine, r.trace_id)
    assert not [k for k in attrs if "harness" in k]


def test_omitted_fields_keep_no_state():
    amund.init(api_key=_KEY, endpoint=_DEAD_ENDPOINT)
    assert tracing._harness_defaults is None
    run = amund.run("agent")
    assert run._harness is None


def test_omitted_fields_never_call_the_validator(monkeypatch):
    def boom(**_):
        raise AssertionError("validator must not run when nothing is declared")

    monkeypatch.setattr(tracing, "clean_harness_layer", boom)
    monkeypatch.setattr(tracing, "merge_harness", boom)
    amund.init(api_key=_KEY, endpoint=_DEAD_ENDPOINT)
    with amund.run("agent"):
        pass
    assert _harness_of(_trace()) == {}


# ---------------------------------------------------------------------------
# Valid values, client config and run overrides
# ---------------------------------------------------------------------------


def test_valid_run_values_land_on_the_json_payload():
    with amund.run(
        "agent",
        harness_hash=A,
        harness_version="support-bot@2026.11.26-r3",
        harness_channels_changed=["prompt", "skill"],
        harness_channels_incomplete=False,
        harness_parent_hash=B,
        harness_run_role="probe",
    ):
        pass
    assert _harness_of(_trace()) == {
        "harness_hash": A,
        "harness_version": "support-bot@2026.11.26-r3",
        "harness_channels_changed": ["prompt", "skill"],
        "harness_channels_incomplete": False,
        "harness_parent_hash": B,
        "harness_run_role": "probe",
    }


def test_client_config_is_the_default_for_every_trace():
    amund.init(
        api_key=_KEY,
        endpoint=_DEAD_ENDPOINT,
        harness_hash=A,
        harness_version="v1",
        harness_run_role="production",
    )
    with amund.run("agent"):
        pass
    assert _harness_of(_trace()) == {
        "harness_hash": A,
        "harness_version": "v1",
        "harness_run_role": "production",
    }


def test_run_overrides_client_field_by_field():
    amund.init(
        api_key=_KEY,
        endpoint=_DEAD_ENDPOINT,
        harness_hash=A,
        harness_version="client",
        harness_run_role="production",
    )
    with amund.run("agent", harness_hash=B, harness_run_role="probe"):
        pass
    assert _harness_of(_trace()) == {
        "harness_hash": B,
        "harness_version": "client",
        "harness_run_role": "probe",
    }


def test_run_dependent_field_uses_the_client_hash():
    amund.init(api_key=_KEY, endpoint=_DEAD_ENDPOINT, harness_hash=A)
    with amund.run("agent", harness_channels_changed=["memory"], harness_parent_hash=B):
        pass
    assert _harness_of(_trace()) == {
        "harness_hash": A,
        "harness_channels_changed": ["memory"],
        "harness_parent_hash": B,
    }


def test_empty_channels_list_is_kept_as_an_assertion_of_no_change():
    with amund.run("agent", harness_hash=A, harness_channels_changed=[]):
        pass
    assert _harness_of(_trace()) == {
        "harness_hash": A,
        "harness_channels_changed": [],
    }


def test_trace_agent_decorator_uses_the_client_default():
    amund.init(api_key=_KEY, endpoint=_DEAD_ENDPOINT, harness_hash=A)

    @amund.trace_agent("decorated")
    def go():
        return 1

    assert go() == 1
    assert _harness_of(_trace()) == {"harness_hash": A}


# ---------------------------------------------------------------------------
# Invalid input: dropped, logged once, never raised
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        "A" * 64,  # uppercase hex
        "a" * 63,
        "a" * 65,
        "sha256:" + "a" * 64,
        "g" * 64,
        "",
        123,
        b"a" * 64,
        ["a" * 64],
    ],
)
def test_bad_hash_is_dropped_with_dependents(bad, caplog):
    with (
        caplog.at_level(logging.DEBUG, logger="amund"),
        amund.run(
            "agent",
            harness_hash=bad,
            harness_version="keeps",
            harness_channels_changed=["prompt"],
            harness_channels_incomplete=True,
            harness_parent_hash=B,
        ),
    ):
        pass
    # harness_version / run_role do not require a hash.
    assert _harness_of(_trace()) == {"harness_version": "keeps"}
    assert "harness_hash dropped" in caplog.text
    assert (
        "harness_channels_changed dropped: requires a valid harness_hash" in caplog.text
    )


def test_unknown_channels_dropped_valid_kept_deduped_and_incomplete_set(caplog):
    channels = ["prompt", "bogus", "skill", "prompt", 7, None]
    with (
        caplog.at_level(logging.DEBUG, logger="amund"),
        amund.run("agent", harness_hash=A, harness_channels_changed=channels),  # type: ignore[arg-type]
    ):
        pass
    got = _harness_of(_trace())
    assert got["harness_channels_changed"] == ["prompt", "skill"]
    assert got["harness_channels_incomplete"] is True
    # one debug line per dropped item (bogus, 7, None); the duplicate is silent
    assert caplog.text.count("harness_channels_changed item dropped") == 3


def test_dropped_channel_forces_incomplete_even_if_caller_said_false():
    with amund.run(
        "agent",
        harness_hash=A,
        harness_channels_changed=["nope"],
        harness_channels_incomplete=False,
    ):
        pass
    got = _harness_of(_trace())
    assert got["harness_channels_changed"] == []
    assert got["harness_channels_incomplete"] is True


def test_duplicates_alone_do_not_set_incomplete():
    with amund.run(
        "agent", harness_hash=A, harness_channels_changed=["plugin", "plugin"]
    ):
        pass
    got = _harness_of(_trace())
    assert got["harness_channels_changed"] == ["plugin"]
    assert "harness_channels_incomplete" not in got


@pytest.mark.parametrize("bad", ["prompt", b"prompt", 5, {"prompt": 1}, object()])
def test_non_list_channels_drop_the_whole_field(bad):
    with amund.run("agent", harness_hash=A, harness_channels_changed=bad):  # type: ignore[arg-type]
        pass
    assert _harness_of(_trace()) == {"harness_hash": A}


@pytest.mark.parametrize("bad", ["admin", "PROBE", "", 1, True, ["probe"]])
def test_bad_run_role_is_dropped(bad):
    with amund.run("agent", harness_hash=A, harness_run_role=bad):  # type: ignore[arg-type]
        pass
    assert _harness_of(_trace()) == {"harness_hash": A}


@pytest.mark.parametrize("bad", ["", "x" * 129, 5, b"v", ["v"]])
def test_bad_version_is_dropped(bad):
    with amund.run("agent", harness_hash=A, harness_version=bad):  # type: ignore[arg-type]
        pass
    assert _harness_of(_trace()) == {"harness_hash": A}


def test_version_boundary_lengths_are_valid():
    with amund.run("agent", harness_version="x" * 128):
        pass
    assert _harness_of(_trace()) == {"harness_version": "x" * 128}


@pytest.mark.parametrize("bad", ["true", 1, 0, None.__class__])
def test_non_bool_incomplete_is_dropped(bad):
    with amund.run("agent", harness_hash=A, harness_channels_incomplete=bad):  # type: ignore[arg-type]
        pass
    assert _harness_of(_trace()) == {"harness_hash": A}


def test_dependent_fields_without_any_hash_are_dropped():
    with amund.run(
        "agent",
        harness_channels_changed=["prompt"],
        harness_channels_incomplete=True,
        harness_parent_hash=B,
        harness_run_role="probe",
    ):
        pass
    assert _harness_of(_trace()) == {"harness_run_role": "probe"}


def test_bad_run_hash_falls_back_to_the_valid_client_hash():
    amund.init(api_key=_KEY, endpoint=_DEAD_ENDPOINT, harness_hash=A)
    with amund.run("agent", harness_hash="nope", harness_channels_changed=["skill"]):
        pass
    assert _harness_of(_trace()) == {
        "harness_hash": A,
        "harness_channels_changed": ["skill"],
    }


def test_invalid_client_config_never_raises_from_init():
    amund.init(
        api_key=_KEY,
        endpoint=_DEAD_ENDPOINT,
        harness_hash=object(),  # type: ignore[arg-type]
        harness_channels_changed=5,  # type: ignore[arg-type]
        harness_run_role="nope",
    )
    with amund.run("agent"):
        pass
    assert _harness_of(_trace()) == {}


def test_hostile_objects_never_escape():
    class Boom:
        def __hash__(self) -> int:
            raise RuntimeError("hash exploded")

        def __eq__(self, other: object) -> bool:
            raise RuntimeError("eq exploded")

        def __iter__(self):
            raise RuntimeError("iter exploded")

        def __str__(self) -> str:
            raise RuntimeError("str exploded")

    with amund.run(
        "agent",
        harness_hash=Boom(),  # type: ignore[arg-type]
        harness_version=Boom(),  # type: ignore[arg-type]
        harness_channels_changed=[Boom()],  # type: ignore[list-item]
        harness_run_role=Boom(),  # type: ignore[arg-type]
    ):
        pass
    assert _harness_of(_trace()).get("harness_hash") is None


# ---------------------------------------------------------------------------
# OTLP: exact keys and types, root span only, no morse.* keys
# ---------------------------------------------------------------------------


def test_otlp_keys_and_types_are_exact(engine):
    with (
        amund.run(
            "agent",
            harness_hash=A,
            harness_version="r3",
            harness_channels_changed=["prompt", "guardrail"],
            harness_channels_incomplete=True,
            harness_parent_hash=B,
            harness_run_role="probe",
        ) as r,
        amund.span("child"),
    ):
        pass
    attrs = _root_attrs(engine, r.trace_id)
    harness_attrs = {k: v for k, v in attrs.items() if "harness" in k}
    assert set(harness_attrs) == {
        "amund.harness.hash",
        "amund.harness.version",
        "amund.harness.channels_changed",
        "amund.harness.channels_incomplete",
        "amund.harness.parent_hash",
        "amund.harness.run_role",
    }
    assert harness_attrs["amund.harness.hash"] == A
    assert harness_attrs["amund.harness.version"] == "r3"
    assert list(harness_attrs["amund.harness.channels_changed"]) == [
        "prompt",
        "guardrail",
    ]
    assert harness_attrs["amund.harness.channels_incomplete"] is True
    assert harness_attrs["amund.harness.parent_hash"] == B
    assert harness_attrs["amund.harness.run_role"] == "probe"
    assert not [k for k in attrs if k.startswith("morse.")]


def test_otlp_only_declared_fields_are_emitted(engine):
    with amund.run("agent", harness_hash=A) as r:
        pass
    attrs = {k: v for k, v in _root_attrs(engine, r.trace_id).items() if "harness" in k}
    assert attrs == {"amund.harness.hash": A}


def test_otlp_harness_is_root_span_only(engine):
    with amund.run("agent", harness_hash=A) as r, amund.span("child"):
        pass
    engine.provider.force_flush()
    tid = int(r.trace_id, 16)
    spans = [s for s in engine.recorder.spans if s.context.trace_id == tid]
    carriers = [s for s in spans if "amund.harness.hash" in dict(s.attributes)]
    assert len(spans) >= 2
    assert len(carriers) == 1


# ---------------------------------------------------------------------------
# Schema validation of the emitted payload
# ---------------------------------------------------------------------------


def test_emitted_payload_validates_against_the_frozen_schema():
    validator = _validator()
    cases = [
        {},
        {"harness_hash": A},
        {
            "harness_hash": A,
            "harness_version": "v",
            "harness_channels_changed": ["prompt", "skill", "plugin"],
            "harness_channels_incomplete": True,
            "harness_parent_hash": B,
            "harness_run_role": "probe",
        },
        {"harness_run_role": "production"},
    ]
    for case in cases:
        with amund.run("agent", **case):
            pass
        assert not list(validator.iter_errors(_harness_of(_trace()))), case


def test_sanitised_garbage_always_validates_against_the_schema():
    validator = _validator()
    garbage = [
        {"harness_hash": "x", "harness_channels_changed": ["a", "prompt"]},
        {"harness_channels_changed": ["prompt"], "harness_parent_hash": C},
        {"harness_hash": A, "harness_channels_changed": ["prompt", "prompt", "z"]},
        {"harness_hash": A, "harness_run_role": "nope", "harness_version": ""},
    ]
    for kwargs in garbage:
        with amund.run("agent", **kwargs):
            pass
        assert not list(validator.iter_errors(_harness_of(_trace()))), kwargs


# ---------------------------------------------------------------------------
# Batch transport / byte-size compatibility
# ---------------------------------------------------------------------------


def test_harness_fields_count_toward_the_serialised_payload_and_stay_json():
    with amund.run("agent"):
        pass
    plain = len(json.dumps(tracing._get_last_trace()))
    with amund.run(
        "agent",
        harness_hash=A,
        harness_version="v",
        harness_channels_changed=["prompt"],
        harness_parent_hash=B,
    ):
        pass
    payload = tracing._get_last_trace()
    assert len(json.dumps(payload)) > plain  # visible to any byte accounting
    assert json.loads(json.dumps(payload))["trace"]["harness_hash"] == A


def test_otlp_batch_with_harness_attrs_exports_every_trace(engine):
    ids = []
    for i in range(20):
        with (
            amund.run(
                "agent", harness_hash=A, harness_channels_changed=["prompt"]
            ) as r,
            amund.span(f"child-{i}"),
        ):
            pass
        ids.append(r.trace_id)
    for trace_id in ids:
        assert _root_attrs(engine, trace_id)["amund.harness.hash"] == A


# ---------------------------------------------------------------------------
# compute_harness_hash
# ---------------------------------------------------------------------------


def _write_tree(root: Path, files: dict[str, bytes]) -> None:
    for rel, data in files.items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)


def test_compute_harness_hash_reproduces_the_pinned_vector(tmp_path):
    _write_tree(tmp_path, FILE_SET_VECTOR_FILES)
    assert amund.compute_harness_hash(tmp_path) == FILE_SET_VECTOR
    assert amund.compute_harness_hash([tmp_path]) == FILE_SET_VECTOR
    assert amund.compute_harness_hash(tmp_path, root=tmp_path) == FILE_SET_VECTOR


def test_compute_harness_hash_files_and_dirs_mix_with_explicit_root(tmp_path):
    _write_tree(tmp_path, FILE_SET_VECTOR_FILES)
    got = amund.compute_harness_hash(
        [
            tmp_path / "prompts",
            tmp_path / "skills/refund/SKILL.md",
            tmp_path / "scripts/check.py",
        ],
        root=tmp_path,
    )
    assert got == FILE_SET_VECTOR


def test_compute_harness_hash_default_root_is_the_common_parent(tmp_path):
    _write_tree(tmp_path, FILE_SET_VECTOR_FILES)
    got = amund.compute_harness_hash(
        [tmp_path / "scripts", tmp_path / "prompts", tmp_path / "skills"]
    )
    assert got == FILE_SET_VECTOR


def test_compute_harness_hash_empty_set_vector(tmp_path):
    assert amund.compute_harness_hash([]) == EMPTY_SET_VECTOR
    assert amund.compute_harness_hash(tmp_path) == EMPTY_SET_VECTOR  # empty dir


def test_compute_harness_hash_is_order_independent_and_byte_exact(tmp_path):
    _write_tree(tmp_path, FILE_SET_VECTOR_FILES)
    files = sorted(p for p in tmp_path.rglob("*") if p.is_file())
    assert (
        amund.compute_harness_hash(files, root=tmp_path)
        == amund.compute_harness_hash(reversed(files), root=tmp_path)
        == FILE_SET_VECTOR
    )
    (tmp_path / "skills/refund/SKILL.md").write_bytes(b"# Refund\nSteps\n")
    assert amund.compute_harness_hash(tmp_path) != FILE_SET_VECTOR


def test_compute_harness_hash_overlapping_inputs_count_each_file_once(tmp_path):
    _write_tree(tmp_path, FILE_SET_VECTOR_FILES)
    got = amund.compute_harness_hash(
        [tmp_path, tmp_path / "prompts/system.md"], root=tmp_path
    )
    assert got == FILE_SET_VECTOR


def test_compute_harness_hash_sorts_by_utf8_bytes_with_nfc_paths(tmp_path):
    names = {"é.md": b"1", "z.md": b"2", "a/b.md": b"3", "日本.md": b"4"}
    nfd = unicodedata.normalize("NFD", "é.md")
    _write_tree(
        tmp_path, {**{k: v for k, v in names.items() if k != "é.md"}, nfd: b"1"}
    )
    lines = []
    for rel, data in names.items():
        p = unicodedata.normalize("NFC", rel)
        lines.append((p.encode(), f"{p}\t{hashlib.sha256(data).hexdigest()}\n"))
    lines.sort()
    expected = hashlib.sha256("".join(line for _, line in lines).encode()).hexdigest()
    assert amund.compute_harness_hash(tmp_path) == expected


@pytest.mark.skipif(os.name == "nt", reason="symlinks")
def test_compute_harness_hash_skips_symlinks_outside_root_and_never_follows_dirs(
    tmp_path,
):
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    _write_tree(root, FILE_SET_VECTOR_FILES)
    _write_tree(outside, {"secret.txt": b"nope"})
    (root / "evil-file").symlink_to(outside / "secret.txt")
    (root / "evil-dir").symlink_to(outside, target_is_directory=True)
    (root / "loop").symlink_to(root, target_is_directory=True)
    assert amund.compute_harness_hash(root) == FILE_SET_VECTOR


@pytest.mark.skipif(os.name == "nt", reason="symlinks")
def test_compute_harness_hash_includes_symlink_to_file_inside_root(tmp_path):
    _write_tree(tmp_path, {"a.md": b"x"})
    (tmp_path / "b.md").symlink_to(tmp_path / "a.md")
    sha = hashlib.sha256(b"x").hexdigest()
    expected = hashlib.sha256(f"a.md\t{sha}\nb.md\t{sha}\n".encode()).hexdigest()
    assert amund.compute_harness_hash(tmp_path) == expected


def test_compute_harness_hash_result_is_a_valid_harness_hash(tmp_path):
    _write_tree(tmp_path, FILE_SET_VECTOR_FILES)
    value = amund.compute_harness_hash(tmp_path)
    with amund.run("agent", harness_hash=value):
        pass
    assert _harness_of(_trace()) == {"harness_hash": value}


def test_compute_harness_hash_errors_are_explicit_not_silent(tmp_path):
    with pytest.raises(FileNotFoundError):
        amund.compute_harness_hash(tmp_path / "missing")
    other = tmp_path / "other"
    other.mkdir()
    (other / "f").write_bytes(b"1")
    inner = tmp_path / "inner"
    inner.mkdir()
    with pytest.raises(ValueError):
        amund.compute_harness_hash(other / "f", root=inner)


def test_module_constants_match_the_vendored_schema():
    schema = json.loads(SCHEMA_PATH.read_text())
    defs = schema["$defs"]
    assert set(defs["harness_channel"]["enum"]) == harness.HARNESS_CHANNELS
    props = defs["harness_fields"]["properties"]
    assert set(props["harness_run_role"]["enum"]) == harness.HARNESS_RUN_ROLES
    assert tuple(props) == harness.HARNESS_FIELDS == FIELDS
    assert props["harness_version"]["maxLength"] == harness.MAX_HARNESS_VERSION_LEN
