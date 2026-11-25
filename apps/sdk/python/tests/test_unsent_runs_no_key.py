"""MHQ-1787 — a run opened with no API key says so, loudly, once, at exit.

With no ``MORSE_API_KEY`` and no ``init()``, ``with amund.run(...)`` keeps
returning a real-looking trace id (unchanged on purpose) while nothing is sent.
The first-use "SDK calls are silent no-ops" warning is easy to miss, so a single
warning at interpreter exit names how many runs were not sent. No new public
attribute exists; the count is internal to diagnostics.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

from amund._internal import diagnostics

_SRC = str(Path(__file__).parent.parent / "src")
_WARNING = "run(s) not sent: no API key (set AMUND_API_KEY or call init())"

_CHILD = """
import sys
sys.path.insert(0, {src!r})
import amund

for i in range({runs}):
    with amund.run("x") as r:
        print("trace_id", r.trace_id)
"""


def _run_child(runs: int, **env_overrides: str) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("MORSE_")}
    env.update(env_overrides)
    return subprocess.run(  # noqa: S603
        [sys.executable, "-c", _CHILD.format(src=_SRC, runs=runs)],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def test_unsent_runs_warn_once_at_exit_and_keep_trace_id():
    proc = _run_child(3)
    assert proc.returncode == 0
    # trace_id semantics are unchanged: a real-looking id is still returned.
    ids = [
        ln.split()[1] for ln in proc.stdout.splitlines() if ln.startswith("trace_id")
    ]
    assert len(ids) == 3
    assert all(len(i) == 32 for i in ids)
    assert proc.stderr.count(_WARNING) == 1
    assert "3 run(s) not sent" in proc.stderr


def test_no_exit_warning_when_disabled_on_purpose():
    proc = _run_child(2, MORSE_DISABLED="1")
    assert proc.returncode == 0
    assert "not sent: no API key" not in proc.stderr


def test_no_exit_warning_when_a_key_is_set():
    proc = _run_child(
        1,
        MORSE_API_KEY="test-key",  # pragma: allowlist secret
        MORSE_ENDPOINT="http://localhost:0",
    )
    assert proc.returncode == 0
    assert "not sent: no API key" not in proc.stderr


def test_warning_helper_counts_once_and_resets(caplog):
    diagnostics._reset_for_testing()
    diagnostics.record_unsent_run_no_key()
    diagnostics.record_unsent_run_no_key()
    with caplog.at_level(logging.WARNING, logger="amund"):
        diagnostics.warn_unsent_runs_at_exit()
        diagnostics.warn_unsent_runs_at_exit()
    msgs = [r.getMessage() for r in caplog.records if "not sent" in r.getMessage()]
    assert msgs == [
        "amund: 2 run(s) not sent: no API key (set AMUND_API_KEY or call init())"
    ]
    diagnostics._reset_for_testing()
