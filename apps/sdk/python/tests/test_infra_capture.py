"""Tests for ADR-019 F2.1 — in-process DB auto-capture.

Uses sqlite3 (stdlib, always available) as the driver under test. The
connect-factory wrapping approach is driver-agnostic, so sqlite3 coverage
exercises the same code path psycopg would.
"""

import sqlite3

import pytest

from morse_ai.infra_capture import (
    disable_db_capture,
    enable_db_capture,
)
from morse_ai.tracing import _get_last_trace, run


@pytest.fixture(autouse=True)
def _restore():
    yield
    disable_db_capture()


def _db_spans(trace: dict) -> list[dict]:
    return [s for s in trace["spans"] if s["type"] == "db"]


def test_enable_wraps_sqlite3():
    wrapped = enable_db_capture()
    assert "sqlite3" in wrapped
    # Idempotent.
    assert "sqlite3" in enable_db_capture()


def test_db_query_inside_trace_emits_db_span():
    enable_db_capture()
    with run("agent-with-db"):
        conn = sqlite3.connect(":memory:")
        cur = conn.cursor()
        cur.execute("CREATE TABLE t (id INTEGER, name TEXT)")
        cur.execute("INSERT INTO t VALUES (1, 'a')")
        cur.execute("SELECT * FROM t")
        cur.fetchall()
        conn.close()

    trace = _get_last_trace()
    db = _db_spans(trace)
    verbs = [s["name"] for s in db]
    assert "CREATE" in verbs
    assert "INSERT" in verbs
    assert "SELECT" in verbs


def test_db_span_captures_statement_text():
    """Q1: the parameterized SQL is stored on the span's input_data."""
    enable_db_capture()
    with run("agent-stmt"):
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE u (id INTEGER, email TEXT)")
        # Parameterized query — the bound value (the email) is in params, NOT
        # in the statement string, so it must NOT appear in the captured span.
        conn.execute("SELECT * FROM u WHERE email = ?", ("secret@example.com",))
        conn.close()

    trace = _get_last_trace()
    select = next(s for s in _db_spans(trace) if s["name"] == "SELECT")
    # input_data is a dict (backend SpanIn requires dict) with the parameterized SQL.
    assert select["input_data"] == {"db.statement": "SELECT * FROM u WHERE email = ?"}
    assert "secret@example.com" not in str(select["input_data"])  # PII-safe


def test_db_statement_truncated_when_huge():
    from morse_ai.infra_capture import _MAX_STATEMENT_LEN, _safe_statement

    huge = "SELECT * FROM t WHERE id IN (" + ",".join(["?"] * 5000) + ")"
    out = _safe_statement(huge)
    assert out.endswith("…[truncated]")
    assert len(out) <= _MAX_STATEMENT_LEN + 20


def test_connection_execute_shortcut_is_captured():
    enable_db_capture()
    with run("agent-shortcut"):
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE x (a INTEGER)")  # Connection.execute shortcut
        conn.close()

    trace = _get_last_trace()
    assert any(s["name"] == "CREATE" for s in _db_spans(trace))


def test_no_db_span_when_no_active_trace():
    enable_db_capture()
    # Outside any run() — must be a transparent passthrough, no error, no span.
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE y (a INTEGER)")
    conn.execute("INSERT INTO y VALUES (1)")
    rows = conn.execute("SELECT a FROM y").fetchall()
    conn.close()
    assert rows == [(1,)]  # query still works correctly


def test_cursor_iteration_and_results_preserved():
    """The proxy must not break normal cursor usage (iteration, fetch)."""
    enable_db_capture()
    with run("agent-iter"):
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE z (a INTEGER)")
        conn.executemany("INSERT INTO z VALUES (?)", [(1,), (2,), (3,)])
        cur = conn.cursor()
        cur.execute("SELECT a FROM z ORDER BY a")
        collected = [row[0] for row in cur]  # exercises __iter__
        conn.close()

    assert collected == [1, 2, 3]
    trace = _get_last_trace()
    verbs = [s["name"] for s in _db_spans(trace)]
    assert "INSERT" in verbs  # executemany captured
    assert "SELECT" in verbs


def test_disable_restores_original_connect():
    enable_db_capture()
    assert sqlite3.connect.__name__ == "_connect"
    disable_db_capture()
    # Original restored — sqlite3.connect is the builtin again.
    assert sqlite3.connect.__name__ != "_connect"


def test_context_manager_connection_still_commits():
    """sqlite3 connection context manager (auto-commit) must keep working."""
    enable_db_capture()
    with run("agent-ctx"):
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE c (a INTEGER)")
        with conn:  # exercises _TracingConnection.__enter__/__exit__
            conn.execute("INSERT INTO c VALUES (42)")
        rows = conn.execute("SELECT a FROM c").fetchall()
        conn.close()
    assert rows == [(42,)]
