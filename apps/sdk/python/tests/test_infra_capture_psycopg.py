"""MHQ-1357 — DB capture keeps psycopg 3 connections real ``psycopg.Connection``s.

The old capture returned a ``_TracingConnection`` proxy from
``psycopg.connect``; LangGraph's ``PostgresSaver`` (and anything else that
type-checks the connection) rejected it with "Invalid connection type", so
checkpoints silently stopped persisting once DB capture was on (default-on
since 0.7.0). psycopg 3 is now traced through its own ``cursor_factory``.

The live tests need a Postgres: set ``MORSE_TEST_PG_DSN`` (e.g. the local
docker-compose database). They skip otherwise.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

psycopg = pytest.importorskip("psycopg")

from amund import infra_capture, tracing  # noqa: E402

_DSN = os.environ.get("MORSE_TEST_PG_DSN")
live = pytest.mark.skipif(not _DSN, reason="MORSE_TEST_PG_DSN not set")


@pytest.fixture
def capture():
    infra_capture.disable_db_capture()
    infra_capture.enable_db_capture()
    token = tracing._active_trace.set(None)
    try:
        yield
    finally:
        infra_capture.disable_db_capture()
        tracing._active_trace.reset(token)


@pytest.mark.req("MHQ-1357")
def test_traced_cursor_factory_is_a_psycopg_cursor_subclass():
    conn = SimpleNamespace(cursor_factory=psycopg.Cursor)
    out = infra_capture._trace_psycopg_connection(conn)
    assert out is conn
    assert issubclass(conn.cursor_factory, psycopg.Cursor)
    assert conn.cursor_factory is not psycopg.Cursor
    # Idempotent: wrapping twice does not stack subclasses.
    first = conn.cursor_factory
    infra_capture._trace_psycopg_connection(conn)
    assert conn.cursor_factory is first


@live
@pytest.mark.req("MHQ-1357")
def test_connect_returns_a_real_connection_and_still_records_db_spans(capture):
    conn = psycopg.connect(_DSN)
    try:
        assert isinstance(conn, psycopg.Connection)
        with tracing.run("pg-capture"):
            conn.execute("SELECT 1").fetchone()
            with conn.cursor() as cur:
                cur.execute("SELECT 2")
                assert cur.fetchone() == (2,)
    finally:
        conn.close()
    trace = tracing._get_last_trace()
    db = [s for s in trace["spans"] if s["type"] == "db"]
    assert [s["name"] for s in db] == ["SELECT", "SELECT"]


@live
@pytest.mark.req("MHQ-1357")
def test_langgraph_postgres_saver_accepts_the_captured_connection(capture):
    pg = pytest.importorskip("langgraph.checkpoint.postgres")
    from psycopg.rows import dict_row

    conn = psycopg.connect(_DSN, autocommit=True, row_factory=dict_row)
    try:
        saver = pg.PostgresSaver(conn)  # raised "Invalid connection type" before
        saver.setup()
    finally:
        conn.close()
