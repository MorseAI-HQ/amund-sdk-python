"""MHQ-1537 / MHQ-1506 — the zero-code import hook, run in a fresh interpreter.

Each test runs a child Python so ``sys.modules`` starts clean: the import
hook only matters for modules imported *after* ``morse_ai.init()``.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest


def _run(code: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env.update(
        {
            "MORSE_API_KEY": "av_test_import_hook",  # pragma: allowlist secret
            "MORSE_ENDPOINT": "http://127.0.0.1:9/api/v1/traces/batch",
        }
    )
    env.pop("MORSE_DISABLE_AUTO_DETECT", None)
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )


@pytest.mark.req("MHQ-1537")
def test_langgraph_adapter_installs_inside_the_import_from_two_threads() -> None:
    """Two threads import LangGraph and langchain_core at once after init().

    Before the fix ``langgraph`` (a namespace package) could not be wrapped, so
    a background thread called ``install()`` while the importing thread was
    still mid-import: the patch landed after the import statement returned,
    and under concurrent imports it deadlocked on a ``langchain_core`` module
    lock. Now the adapter installs at the end of ``langgraph.pregel``'s body,
    in the importing thread, before ``from langgraph.graph import ...``
    returns.
    """
    proc = _run(
        """
        import threading
        import morse_ai
        from morse_ai._internal import auto_detect

        installs = []
        _real_install_one = auto_detect._install_one

        def recording_install_one(path):
            installs.append((path, threading.current_thread().name))
            return _real_install_one(path)

        auto_detect._install_one = recording_install_one
        morse_ai.init()

        installed_at_return = {}
        barrier = threading.Barrier(2)

        def import_graph():
            barrier.wait()
            from langgraph.graph import StateGraph  # noqa: F401
            import langgraph
            installed_at_return["graph"] = bool(
                getattr(langgraph, "_morse_ai_installed", False)
            )

        def import_core():
            barrier.wait()
            import langchain_core.messages.utils  # noqa: F401
            from langchain_core.callbacks.manager import CallbackManager  # noqa: F401

        threads = [
            threading.Thread(target=f, name=f.__name__)
            for f in (import_graph, import_core)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(60)
        assert not any(t.is_alive() for t in threads), "import hung"
        lg = [name for path, name in installs if path.endswith(".langgraph")]
        print("INSTALLED_AT_RETURN", installed_at_return.get("graph"))
        print("LANGGRAPH_INSTALLED_BY", lg)
        """
    )
    assert proc.returncode == 0, proc.stderr
    assert "INSTALLED_AT_RETURN True" in proc.stdout, proc.stdout + proc.stderr
    # Installed synchronously by the thread doing the import, never by a
    # background poller racing it.
    assert "LANGGRAPH_INSTALLED_BY ['import_graph']" in proc.stdout, proc.stdout
    assert "_DeadlockError" not in proc.stderr
    assert "internal error" not in proc.stderr


@pytest.mark.req("MHQ-1537")
def test_langgraph_imported_before_init_installs_immediately() -> None:
    proc = _run(
        """
        from langgraph.graph import StateGraph  # noqa: F401
        import langgraph, morse_ai
        morse_ai.init()
        print("INSTALLED", getattr(langgraph, "_morse_ai_installed", False))
        """
    )
    assert proc.returncode == 0, proc.stderr
    assert "INSTALLED True" in proc.stdout


@pytest.mark.req("MHQ-1506")
def test_zero_code_setup_patches_raw_openai_imported_after_init() -> None:
    """``MORSE_API_KEY`` + ``init()`` now covers a raw OpenAI client."""
    proc = _run(
        """
        import morse_ai
        morse_ai.init()
        import openai
        from openai.resources.chat.completions import Completions
        from openai.resources.responses import Responses
        print("INSTALLED", getattr(openai, "_morse_ai_installed", False))
        print("CHAT", hasattr(Completions.create, "__wrapped__"))
        print("RESPONSES", hasattr(Responses.create, "__wrapped__"))
        """
    )
    assert proc.returncode == 0, proc.stderr
    assert "INSTALLED True" in proc.stdout
    assert "CHAT True" in proc.stdout
    assert "RESPONSES True" in proc.stdout


@pytest.mark.req("MHQ-1506")
@pytest.mark.parametrize("before_init", [False, True])
def test_zero_code_setup_patches_google_genai(before_init: bool) -> None:
    """``google.genai`` is traced whether it is imported before or after init()."""
    pytest.importorskip("google.genai")
    imp = "from google import genai\n"
    proc = _run(
        (imp if before_init else "")
        + "import morse_ai\nmorse_ai.init()\n"
        + ("" if before_init else imp)
        + "from google.genai.models import AsyncModels, Models\n"
        + 'print("INSTALLED", getattr(genai, "_morse_ai_installed", False))\n'
        + 'print("SYNC", hasattr(Models._generate_content, "__wrapped__"))\n'
        + "stream = AsyncModels._generate_content_stream\n"
        + 'print("ASYNC", hasattr(stream, "__wrapped__"))\n'
    )
    assert proc.returncode == 0, proc.stderr
    assert "INSTALLED True" in proc.stdout
    assert "SYNC True" in proc.stdout
    assert "ASYNC True" in proc.stdout
