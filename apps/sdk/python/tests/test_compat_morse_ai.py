"""The ``morse-ai`` compatibility distribution (compat/morse-ai).

After the Morse -> Amund rename the SDK ships as ``amund``. The ``morse-ai``
wheel only contains a ``morse_ai`` alias package. These tests prove that every
old import path keeps working and resolves to the *same* objects as ``amund``
(so isinstance checks, singletons and mock.patch targets are shared), and that
the alias warns exactly once.

Each scenario runs in a fresh interpreter: the alias rewrites ``sys.modules``
and ``sys.meta_path``, and the deprecation warning fires only on first import.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import tomllib
from pathlib import Path

import pytest

_SDK_ROOT = Path(__file__).resolve().parent.parent
_SHIM_ROOT = _SDK_ROOT / "compat" / "morse-ai"
if not (_SHIM_ROOT / "pyproject.toml").exists():  # e.g. running from the sdist
    pytest.skip("compat/morse-ai is not in this tree", allow_module_level=True)

_PYTHONPATH = os.pathsep.join([str(_SHIM_ROOT / "src"), str(_SDK_ROOT / "src")])


def _run(code: str) -> subprocess.CompletedProcess[str]:
    env = {
        k: v for k, v in os.environ.items() if not k.startswith(("AMUND_", "MORSE_"))
    }
    env["PYTHONPATH"] = _PYTHONPATH
    env["MORSE_ENDPOINT"] = "http://localhost:0"
    proc = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return proc


def test_import_morse_ai_is_the_amund_package():
    _run(
        """
        import warnings
        warnings.simplefilter("ignore")
        import morse_ai
        import amund
        assert morse_ai is amund
        assert morse_ai.init is amund.init
        assert morse_ai.run is amund.run
        assert morse_ai.__version__ == amund.__version__
        import sys
        assert sys.modules["morse_ai"] is amund
        """
    )


def test_deprecation_warning_fires_once_and_names_the_new_package():
    proc = _run(
        """
        import warnings
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            import morse_ai
            import morse_ai
            import morse_ai.tracing
            from morse_ai.adapters import openai
            from morse_ai import logging
        deps = [w for w in caught if issubclass(w.category, DeprecationWarning)
                and "amund" in str(w.message)]
        assert len(deps) == 1, [str(w.message) for w in caught]
        msg = str(deps[0].message)
        assert "pip install amund" in msg and "import amund" in msg, msg
        print("WARNED", deps[0].filename)
        """
    )
    # stacklevel points at the importing code, not at the shim itself.
    assert "morse_ai" not in proc.stdout.split("WARNED", 1)[1]


def test_amund_import_alone_does_not_warn():
    _run(
        """
        import warnings
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            import amund
        assert not [w for w in caught if issubclass(w.category, DeprecationWarning)
                    and "morse" in str(w.message).lower()]
        """
    )


def test_old_submodule_paths_resolve_to_the_same_module_objects():
    _run(
        """
        import warnings
        warnings.simplefilter("ignore")
        import sys
        import importlib

        # Loaded eagerly by `import amund` ...
        import morse_ai.tracing, morse_ai.client, morse_ai._internal.env
        import amund.tracing, amund.client, amund._internal.env
        assert morse_ai.tracing is amund.tracing
        assert morse_ai.client is amund.client
        assert morse_ai._internal.env is amund._internal.env

        # ... and lazily-loaded ones, imported through the OLD name first.
        lazy = [
            "adapters", "adapters.base", "adapters.openai", "adapters.anthropic",
            "adapters.langchain", "logging", "infra_capture", "queue_capture",
            "otel_provider", "pricing", "context.decomposer", "data.compat",
            "instrumentation.http_classifier",
        ]
        for sub in lazy:
            old = importlib.import_module("morse_ai." + sub)
            new = importlib.import_module("amund." + sub)
            assert old is new, sub
            assert sys.modules["morse_ai." + sub] is sys.modules["amund." + sub]
            # The real module keeps its real identity.
            assert new.__name__ == "amund." + sub
            assert new.__spec__.name == "amund." + sub, new.__spec__

        # No second copy of any amund source file was executed.
        files = {}
        for name, mod in list(sys.modules.items()):
            f = getattr(mod, "__file__", None)
            if f and ("/amund/" in f):
                assert files.setdefault(f, mod) is mod, (name, f)
        """
    )


def test_from_imports_return_the_same_classes_and_functions():
    _run(
        """
        import warnings
        warnings.simplefilter("ignore")
        from morse_ai.adapters.openai import install as old_install
        from morse_ai.adapters.langchain import MorseCallbackHandler as OldHandler
        from morse_ai.logging import MorseHandler as OldLogHandler
        from morse_ai.spans import SpanData as OldSpanData, SpanType as OldSpanType
        from morse_ai.tracing import TracedRun as OldTracedRun
        from morse_ai import RunRecord as OldRunRecord

        from amund.adapters.openai import install
        from amund.adapters.langchain import MorseCallbackHandler
        from amund.logging import MorseHandler
        from amund.spans import SpanData, SpanType
        from amund.tracing import TracedRun
        from amund import RunRecord

        assert old_install is install
        assert OldHandler is MorseCallbackHandler
        assert OldLogHandler is MorseHandler
        assert OldSpanData is SpanData and OldSpanType is SpanType
        assert OldTracedRun is TracedRun
        assert OldRunRecord is RunRecord
        span = OldSpanData(trace_id="t", type=OldSpanType.TOOL, name="x")
        assert isinstance(span, SpanData)
        """
    )


def test_singleton_state_is_shared():
    _run(
        """
        import warnings
        warnings.simplefilter("ignore")
        import amund, morse_ai
        key = "test-key"  # pragma: allowlist secret
        morse_ai.init(api_key=key, endpoint="http://localhost:0")
        assert amund._client is not None
        assert morse_ai._client is amund._client
        amund.shutdown()
        """
    )


def test_mock_patch_on_old_path_patches_amund():
    _run(
        """
        import warnings
        warnings.simplefilter("ignore")
        from unittest import mock
        import morse_ai
        import amund.tracing

        sentinel = object()
        with mock.patch("morse_ai.tracing.get_active_trace", return_value=sentinel):
            assert amund.tracing.get_active_trace() is sentinel
        assert amund.tracing.get_active_trace() is not sentinel
        """
    )


def test_unknown_old_submodule_still_raises_module_not_found():
    _run(
        """
        import warnings
        warnings.simplefilter("ignore")
        import morse_ai
        try:
            import morse_ai.does_not_exist  # noqa: F401
        except ModuleNotFoundError:
            pass
        else:
            raise AssertionError("expected ModuleNotFoundError")
        """
    )


def test_legacy_env_var_still_initialises_through_the_old_import():
    _run(
        """
        import os, warnings
        warnings.simplefilter("ignore")
        os.environ["MORSE_API_KEY"] = "legacy-key"  # pragma: allowlist secret
        import morse_ai
        from morse_ai._internal import env
        assert env.get_api_key() == "legacy-key"
        os.environ["AMUND_API_KEY"] = "new-key"  # pragma: allowlist secret
        assert env.get_api_key() == "new-key"
        """
    )


def _toml(path: Path) -> dict:
    with path.open("rb") as fh:
        return tomllib.load(fh)


def test_shim_version_and_pins_track_amund():
    """The shim must be released in lockstep: same version, exact amund pin."""
    amund_meta = _toml(_SDK_ROOT / "pyproject.toml")["project"]
    shim_meta = _toml(_SHIM_ROOT / "pyproject.toml")["project"]
    version = amund_meta["version"]

    assert amund_meta["name"] == "amund"
    assert shim_meta["name"] == "morse-ai"
    assert shim_meta["version"] == version
    assert shim_meta["dependencies"] == [f"amund=={version}"]

    # Every amund extra exists on the shim and pins the same version.
    amund_extras = set(amund_meta["optional-dependencies"]) - {"dev"}
    shim_extras = shim_meta["optional-dependencies"]
    assert set(shim_extras) == amund_extras
    for extra, reqs in shim_extras.items():
        assert reqs == [f"amund[{extra}]=={version}"], extra


def test_shim_wheel_ships_only_the_alias_package():
    shim = _toml(_SHIM_ROOT / "pyproject.toml")
    assert shim["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"] == [
        "src/morse_ai"
    ]
    files = sorted(
        p.relative_to(_SHIM_ROOT / "src").as_posix()
        for p in (_SHIM_ROOT / "src").rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
    )
    assert files == ["morse_ai/__init__.py"]


@pytest.mark.parametrize("name", ["amund", "morse_ai"])
def test_both_names_importable_in_process(name):
    """Smoke check in the shared test process too (no subprocess)."""
    import importlib
    import warnings

    sys.path.insert(0, str(_SHIM_ROOT / "src"))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            mod = importlib.import_module(name)
    finally:
        sys.path.remove(str(_SHIM_ROOT / "src"))
    import amund

    assert mod is amund
