"""INFRACAP-01..05 — http/db capture is default-on, bounded and scrubbed."""

import sqlite3

import httpx
import pytest

from amund import infra_capture as ic
from amund.tracing import _get_last_trace, run

_ENV = (
    "MORSE_CAPTURE_INFRA",
    "MORSE_CAPTURE_DB",
    "MORSE_CAPTURE_HTTP",
    "MORSE_CAPTURE_HTTP_EXCLUDE_HOSTS",
)


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for name in _ENV:
        monkeypatch.delenv(name, raising=False)
    ic.disable_db_capture()
    ic.disable_http_capture()
    yield
    ic.disable_db_capture()
    ic.disable_http_capture()


def _spans(kind):
    return [s for s in _get_last_trace()["spans"] if s["type"] == kind]


def _client():
    return httpx.Client(
        transport=httpx.MockTransport(lambda req: httpx.Response(200, json={}))
    )


# --- INFRACAP-01 / 02 ------------------------------------------------------


@pytest.mark.req("INFRACAP-01")
def test_defaults_enable_db_and_http():
    ic.maybe_enable_from_env()
    assert "sqlite3.connect" in ic._originals
    assert "httpx" in ic._http_originals
    with run("agent"):
        sqlite3.connect(":memory:").execute("SELECT 1")
        _client().get("https://example.com/x")
    assert len(_spans("db")) == 1
    assert len(_spans("http")) == 1


@pytest.mark.req("INFRACAP-02")
@pytest.mark.parametrize("value", ["0", "false", "no", "off", "FALSE"])
def test_infra_zero_disables_both(monkeypatch, value):
    monkeypatch.setenv("MORSE_CAPTURE_INFRA", value)
    ic.maybe_enable_from_env()
    assert not ic._originals
    assert not ic._http_originals


@pytest.mark.req("INFRACAP-02")
def test_db_zero_disables_only_db(monkeypatch):
    monkeypatch.setenv("MORSE_CAPTURE_DB", "0")
    ic.maybe_enable_from_env()
    assert not ic._originals
    assert "httpx" in ic._http_originals


@pytest.mark.req("INFRACAP-02")
def test_http_zero_disables_only_http(monkeypatch):
    monkeypatch.setenv("MORSE_CAPTURE_HTTP", "0")
    ic.maybe_enable_from_env()
    assert "sqlite3.connect" in ic._originals
    assert not ic._http_originals


@pytest.mark.req("INFRACAP-02")
def test_specific_flag_wins_over_infra(monkeypatch):
    monkeypatch.setenv("MORSE_CAPTURE_INFRA", "0")
    monkeypatch.setenv("MORSE_CAPTURE_DB", "1")
    ic.maybe_enable_from_env()
    assert "sqlite3.connect" in ic._originals
    assert not ic._http_originals


@pytest.mark.req("INFRACAP-02")
def test_garbage_value_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("MORSE_CAPTURE_INFRA", "maybe")
    ic.maybe_enable_from_env()
    assert "sqlite3.connect" in ic._originals


# --- INFRACAP-03 -----------------------------------------------------------


@pytest.mark.req("INFRACAP-03")
def test_no_trace_no_span_and_calls_still_work():
    ic.maybe_enable_from_env()
    conn = sqlite3.connect(":memory:")
    assert conn.execute("SELECT 41 + 1").fetchone() == (42,)
    assert _client().get("https://example.com/x").status_code == 200


# --- INFRACAP-04 -----------------------------------------------------------


@pytest.mark.req("INFRACAP-04")
def test_db_cap_then_one_summary_span():
    ic.maybe_enable_from_env()
    with run("agent"):
        conn = sqlite3.connect(":memory:")
        for i in range(250):
            conn.execute("SELECT ?", (i,))
    db = _spans("db")
    summaries = [s for s in db if "infra.suppressed_count" in s.get("metadata", {})]
    assert len(db) == 201
    assert len(summaries) == 1
    assert summaries[0]["metadata"]["infra.suppressed_count"] == 50


@pytest.mark.req("INFRACAP-04")
def test_http_cap_then_one_summary_span():
    ic.maybe_enable_from_env()
    with run("agent"):
        client = _client()
        for _ in range(130):
            client.get("https://example.com/x")
    http = _spans("http")
    summaries = [s for s in http if "infra.suppressed_count" in s.get("metadata", {})]
    assert len(http) == 101
    assert summaries[0]["metadata"]["infra.suppressed_count"] == 30


@pytest.mark.req("INFRACAP-04")
def test_caps_are_per_trace():
    ic.maybe_enable_from_env()
    for _ in range(2):
        with run("agent"):
            conn = sqlite3.connect(":memory:")
            for _i in range(150):
                conn.execute("SELECT 1")
        assert len(_spans("db")) == 150


# --- INFRACAP-05 -----------------------------------------------------------


@pytest.mark.req("INFRACAP-05")
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (
            "SELECT * FROM users WHERE email = 'a@b.com' AND id = 42",
            "SELECT * FROM users WHERE email = ? AND id = ?",
        ),
        (
            "SELECT * FROM t WHERE a = %s AND b = ?",
            "SELECT * FROM t WHERE a = %s AND b = ?",
        ),
        ("SELECT $1, :name FROM t1", "SELECT $1, :name FROM t1"),
        (
            "INSERT INTO t VALUES ('it''s secret', 3.14, 0xDEADBEEF)",
            "INSERT INTO t VALUES (?, ?, ?)",
        ),
        ("SELECT $tag$secret body$tag$", "SELECT ?"),
        ("SELECT $$secret body$$", "SELECT ?"),
        ("SELECT 1 -- ssn 123-45-6789\n, 2", "SELECT ? \n, ?"),
        ("SELECT /* tok_abc */ 1", "SELECT  ?"),
        ("SELECT E'a\\'b' , 'c'", "SELECT ? , ?"),
        ("WHERE x IN (1, 2, 3) LIMIT 10", "WHERE x IN (?, ?, ?) LIMIT ?"),
    ],
)
def test_scrub_sql(raw, expected):
    assert ic.scrub_sql(raw) == expected


@pytest.mark.req("INFRACAP-05")
def test_scrub_sql_unterminated_quote_drops_the_tail():
    out = ic.scrub_sql("SELECT 'oops secret-value")
    assert "secret-value" not in out


@pytest.mark.req("INFRACAP-05")
def test_inlined_literal_never_reaches_the_span():
    ic.maybe_enable_from_env()
    with run("agent"):
        conn = sqlite3.connect(":memory:")
        conn.execute("SELECT 'canary@example.com', 4111111111111111")
    stmt = _spans("db")[0]["input_data"]["db.statement"]
    assert "canary" not in stmt
    assert "4111" not in stmt


@pytest.mark.req("INFRACAP-05")
def test_bound_params_never_stored():
    ic.maybe_enable_from_env()
    with run("agent"):
        sqlite3.connect(":memory:").execute("SELECT ?", ("canary@example.com",))
    assert "canary" not in str(_get_last_trace())
