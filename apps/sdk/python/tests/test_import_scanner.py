"""Tests for amund.instrumentation.imports."""

import sys
import types

import pytest

from amund.instrumentation.imports import (
    AVImportWatcher,
    install_import_watcher,
    scan_imported_libraries,
    uninstall_import_watcher,
)


@pytest.mark.req("MEM-18")
class TestScanImportedLibraries:
    def test_returns_dict(self):
        result = scan_imported_libraries()
        assert isinstance(result, dict)

    def test_tracked_lib_present_when_module_imported(self):
        # Inject a fake module for "asyncpg" (tracked) into sys.modules.
        if "asyncpg" not in sys.modules:
            sys.modules["asyncpg"] = types.ModuleType("asyncpg")
        result = scan_imported_libraries()
        assert "asyncpg" in result

    def test_version_is_string(self):
        if "asyncpg" not in sys.modules:
            sys.modules["asyncpg"] = types.ModuleType("asyncpg")
        result = scan_imported_libraries()
        version = result.get("asyncpg")
        assert isinstance(version, str)

    def test_unimported_lib_absent(self):
        # A lib whose module is NOT in sys.modules should not appear.
        lib_mod = "chromadb"
        was_present = lib_mod in sys.modules
        if was_present:
            pytest.skip("chromadb happens to be imported in this env")
        result = scan_imported_libraries()
        assert lib_mod not in result

    def test_anthropic_present_if_imported(self):
        try:
            import anthropic  # noqa: F401
        except ImportError:
            pytest.skip("anthropic not installed")
        result = scan_imported_libraries()
        assert "anthropic" in result


@pytest.mark.req("MEM-18")
class TestAVImportWatcher:
    def teardown_method(self):
        uninstall_import_watcher()

    def test_callback_fires_on_first_import(self):
        fired: list[tuple[str, str]] = []

        def cb(lib: str, version: str) -> None:
            fired.append((lib, version))

        install_import_watcher(cb)

        from amund.instrumentation.imports import _watcher_instance

        assert _watcher_instance is not None
        # Reset seen set so the callback fires for this test.
        _watcher_instance._seen.discard("asyncpg")
        _watcher_instance.find_spec("asyncpg", None)

        assert any(lib == "asyncpg" for lib, _ in fired)

    def test_callback_fires_only_once(self):
        fired: list[str] = []
        install_import_watcher(lambda lib, _v: fired.append(lib))

        from amund.instrumentation.imports import _watcher_instance

        assert _watcher_instance is not None
        _watcher_instance._seen.discard("asyncpg")

        _watcher_instance.find_spec("asyncpg", None)
        _watcher_instance.find_spec("asyncpg", None)

        assert fired.count("asyncpg") == 1

    def test_find_spec_always_returns_none(self):
        install_import_watcher(lambda lib, v: None)
        from amund.instrumentation.imports import _watcher_instance

        assert _watcher_instance is not None
        result = _watcher_instance.find_spec("asyncpg", None)
        assert result is None

    def test_non_tracked_module_ignored(self):
        fired: list[str] = []
        install_import_watcher(lambda lib, _v: fired.append(lib))
        from amund.instrumentation.imports import _watcher_instance

        assert _watcher_instance is not None
        _watcher_instance.find_spec("some_random_lib_xyz", None)
        assert fired == []


@pytest.mark.req("MEM-18")
class TestWatcherLifecycle:
    def teardown_method(self):
        uninstall_import_watcher()

    def test_install_is_idempotent(self):
        cb = lambda lib, v: None  # noqa: E731
        install_import_watcher(cb)
        install_import_watcher(cb)
        watchers = [f for f in sys.meta_path if isinstance(f, AVImportWatcher)]
        assert len(watchers) == 1

    def test_uninstall_is_idempotent(self):
        uninstall_import_watcher()
        uninstall_import_watcher()

    def test_install_then_uninstall(self):
        install_import_watcher(lambda lib, v: None)
        watchers_before = [f for f in sys.meta_path if isinstance(f, AVImportWatcher)]
        assert len(watchers_before) == 1

        uninstall_import_watcher()
        watchers_after = [f for f in sys.meta_path if isinstance(f, AVImportWatcher)]
        assert len(watchers_after) == 0
