"""Phase 6c Wave 2 — auto-detection layer tests.

Verifies:
- morse_ai.init() triggers install_adapters()
- Anthropic adapter install/uninstall is idempotent
- LangGraph wins over generic LangChain when both importable
- MORSE_DISABLE_AUTO_DETECT=1 short-circuits the scan
- Lazy import hook fires when target SDK imported AFTER morse_ai.init()
- Adapter install failures don't propagate (non-invasive)
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

import morse_ai
from morse_ai._internal import auto_detect
from morse_ai.adapters import anthropic as anthropic_adapter


def setup_function() -> None:
    morse_ai._reset_for_testing()


def teardown_function() -> None:
    morse_ai._reset_for_testing()


# ---------------------------------------------------------------------------
# install_adapters integration
# ---------------------------------------------------------------------------


def test_init_calls_install_adapters() -> None:
    """morse_ai.init() should call auto_detect.install_adapters() once."""
    with patch("morse_ai.auto_detect.install_adapters") as mock_install:
        morse_ai.init(api_key="av_test")
        mock_install.assert_called_once()


def test_env_auto_init_calls_install_adapters(monkeypatch: pytest.MonkeyPatch) -> None:
    """Env-based auto-init triggers install_adapters() too."""
    monkeypatch.setenv("MORSE_API_KEY", "av_env_test")
    with patch("morse_ai.auto_detect.install_adapters") as mock_install:
        # First public call triggers env-based auto-init
        morse_ai.track(agent_id="a", run_data={})
        mock_install.assert_called_once()


def test_disable_auto_detect_short_circuits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MORSE_DISABLE_AUTO_DETECT=1 makes install_adapters() a no-op."""
    monkeypatch.setenv("MORSE_DISABLE_AUTO_DETECT", "1")
    # No assertion on side effects — just confirm no-raise and no install.
    auto_detect.install_adapters()
    assert not anthropic_adapter.is_installed()


# ---------------------------------------------------------------------------
# Anthropic adapter install/uninstall idempotency
# ---------------------------------------------------------------------------


def test_anthropic_install_when_anthropic_unavailable() -> None:
    """install() returns False (not raises) when anthropic isn't installed."""
    if "anthropic" in sys.modules:
        pytest.skip("anthropic IS installed in this env — opposite scenario")
    result = anthropic_adapter.install()
    assert result is False


def test_anthropic_install_idempotent_when_available() -> None:
    """When anthropic is installed, install() is idempotent."""
    if "anthropic" not in sys.modules:
        try:
            import anthropic  # noqa: F401
        except ImportError:
            pytest.skip("anthropic not available in this test env")

    first = anthropic_adapter.install()
    second = anthropic_adapter.install()
    third = anthropic_adapter.install()

    assert first is True
    assert second is True
    assert third is True
    assert anthropic_adapter.is_installed() is True

    # Cleanup
    anthropic_adapter.uninstall()
    assert anthropic_adapter.is_installed() is False


def test_anthropic_uninstall_idempotent() -> None:
    """uninstall() is safe to call on a never-installed adapter."""
    anthropic_adapter.uninstall()  # not installed; should be no-op
    anthropic_adapter.uninstall()  # again — still no-op
    assert anthropic_adapter.is_installed() is False


# ---------------------------------------------------------------------------
# Priority rules
# ---------------------------------------------------------------------------


def test_langgraph_wins_over_langchain_when_both_present() -> None:
    """When langgraph is importable, the auto-detect targets include it."""
    fake_spec = MagicMock()
    with patch(
        "morse_ai._internal.auto_detect.importlib.util.find_spec",
        return_value=fake_spec,
    ):
        targets = auto_detect._adapter_targets()
        assert "langgraph" in targets
        assert "langchain" not in targets, (
            "langchain fallback should NOT be included when langgraph is available"
        )


def test_no_langgraph_means_no_langgraph_target() -> None:
    """When langgraph is unavailable, it's not in the target list."""
    with patch(
        "morse_ai._internal.auto_detect.importlib.util.find_spec",
        return_value=None,
    ):
        targets = auto_detect._adapter_targets()
        assert "langgraph" not in targets


# ---------------------------------------------------------------------------
# Non-invasive guarantee for adapter install failures
# ---------------------------------------------------------------------------


def test_install_adapters_absorbs_individual_install_failures() -> None:
    """If one adapter's install() raises, others still install; init() succeeds."""

    def _broken_install() -> bool:
        raise RuntimeError("anthropic install boom")

    with patch.object(anthropic_adapter, "install", side_effect=_broken_install):
        # MUST NOT raise — install_adapters wraps each adapter call.
        auto_detect.install_adapters()


# ---------------------------------------------------------------------------
# Lazy import hook
# ---------------------------------------------------------------------------


def test_lazy_finder_registered_when_target_not_imported() -> None:
    """If a target isn't in sys.modules, a lazy finder is added to sys.meta_path."""
    # Force a state where a target is "not imported" by removing if present
    # and stubbing find_spec to claim no langgraph.
    with patch(
        "morse_ai._internal.auto_detect.importlib.util.find_spec",
        return_value=None,
    ):
        # Remove anthropic from sys.modules temporarily to simulate "not imported"
        was_imported = "anthropic" in sys.modules
        if was_imported:
            saved = sys.modules.pop("anthropic")
        try:
            auto_detect.install_adapters()
            # The finder should be in sys.meta_path now.
            assert auto_detect._finder_installed is True
            assert auto_detect._finder_singleton in sys.meta_path
            assert "anthropic" in auto_detect._lazy_targets
        finally:
            if was_imported:
                sys.modules["anthropic"] = saved
            auto_detect._reset_for_testing()
