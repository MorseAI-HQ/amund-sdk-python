"""MHQ-947: a user value that raises during coercion must not reach host code.

The non-invasive guarantee says telemetry never breaks the application. Cost
dimensions are USER values and the SDK is the one that chooses to `str()` them,
so a bare coercion runs arbitrary customer code — `__str__` / `__repr__` — on
the telemetry path.

Two sites had it unguarded, both confirmed reproducible on 0.5.0rc5:

  set_context(**dims)          raised INTO host code
  run(user_metadata=...)       raised at trace START, taking the whole `with`
                               block down with it

The trigger is not exotic. An ORM object whose `__str__` lazily hits a dead
connection, or a model with a broken `__repr__`, gets there without anyone
being clever.

What these pin is the SHAPE of the fix, not just the absence of a crash: the
guard is per key, so one bad dimension costs that dimension and the rest still
arrive. Wrapping the whole call would have been easier and would have silently
thrown away the customer's good dimensions too.
"""

from __future__ import annotations

import pytest

import morse_ai
from morse_ai._internal import diagnostics
from morse_ai.tracing import _active_trace, coerce_dims


class Detonator:
    """Raises from every path `str()` might take."""

    def __repr__(self) -> str:
        raise RuntimeError("coercion probe: __repr__")

    def __str__(self) -> str:
        raise RuntimeError("coercion probe: __str__")


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    # Deliberately NO MORSE_API_KEY / MORSE_ENDPOINT: with no resolvable
    # target the diagnostics sender is never built, so these tests never start
    # its thread. That matters beyond tidiness — `_reset_for_testing()` sets
    # `_sender = None` but cannot join the thread, which is parked in a 5s
    # sleep, so a test that builds one leaves a live `morse_ai-diagnostics`
    # thread behind and breaks SDKDIAG-01's "a healthy process arms nothing"
    # assertion in whichever file runs next. Observed exactly that.
    #
    # Nothing here needs transmission: `report()` still runs, and these tests
    # are about what coercion does, not where the report lands.
    for var in ("MORSE_API_KEY", "MORSE_ENDPOINT"):
        monkeypatch.delenv(var, raising=False)
    diagnostics._reset_for_testing()
    morse_ai._reset_for_testing()
    yield
    diagnostics._reset_for_testing()
    morse_ai._reset_for_testing()


def test_coerce_dims_drops_only_the_offender():
    out = coerce_dims(
        {"customer_id": "cust-42", "feature": Detonator(), "team": "core"},
        "test.fingerprint",
    )

    assert out == {"customer_id": "cust-42", "team": "core"}


def test_coerce_dims_never_raises_on_an_all_bad_mapping():
    assert coerce_dims({"a": Detonator(), "b": Detonator()}, "test.fingerprint") == {}


def test_coerce_dims_reports_each_failure(monkeypatch):
    seen: list[str] = []
    monkeypatch.setattr(
        diagnostics, "report", lambda exc, fingerprint: seen.append(fingerprint)
    )

    coerce_dims({"bad": Detonator(), "worse": Detonator()}, "test.fingerprint")

    assert seen == ["test.fingerprint", "test.fingerprint"]


def test_set_context_does_not_raise_into_host_code():
    with morse_ai.run("coercion-probe"):
        # Before MHQ-947 this raised RuntimeError straight through.
        morse_ai.set_context(customer_id="cust-42", feature=Detonator(), team="core")

        state = _active_trace.get(None)
        assert state is not None
        assert state.user_metadata == {"customer_id": "cust-42", "team": "core"}


def test_run_user_metadata_does_not_break_trace_start():
    body_ran = False
    # Before MHQ-947 the `with` never opened: coercion happened in __enter__.
    with morse_ai.run(
        "coercion-probe", user_metadata={"customer_id": "cust-7", "bad": Detonator()}
    ):
        body_ran = True
        state = _active_trace.get(None)
        assert state is not None
        assert state.user_metadata == {"customer_id": "cust-7"}

    assert body_ran, "the traced block never executed"
