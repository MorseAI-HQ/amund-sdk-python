"""SDK get_prompt — thin hosted-prompt fetch (openspec prompt-registry-sdk-fetch).

HTTP is mocked with respx; nothing touches the network.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import httpx
import pytest
import respx

import amund
from amund import (
    Prompt,
    PromptAccessError,
    PromptError,
    PromptNotFoundError,
    PromptRateLimitError,
)
from amund import prompts as _prompts

BASE = "https://api.morse.test"
BODY = {
    "name": "triage",
    "version": 3,
    "template_text": "Hello {{user}}",
    "labels": ["production"],
    "etag": '"abc"',
}


@pytest.fixture(autouse=True)
def _client(monkeypatch):
    fake = SimpleNamespace(
        endpoint=f"{BASE}/api/v1/traces/batch",
        _api_key="mk_test_fake",  # pragma: allowlist secret
    )
    monkeypatch.setattr(amund, "_client", fake)
    yield


@pytest.mark.req("SPF-01")
@respx.mock
def test_default_label_production_and_bearer_auth():
    route = respx.get(f"{BASE}/api/v1/prompts/triage").mock(
        return_value=httpx.Response(200, json=BODY, headers={"ETag": '"abc"'})
    )
    p = amund.get_prompt("triage")
    req = route.calls.last.request
    assert dict(req.url.params) == {"label": "production"}
    assert req.headers["authorization"] == "Bearer mk_test_fake"
    assert p == Prompt("triage", 3, "Hello {{user}}", ["production"], '"abc"')
    assert p.not_modified is False


@pytest.mark.req("SPF-01")
@respx.mock
def test_label_selector():
    route = respx.get(f"{BASE}/api/v1/prompts/triage").mock(
        return_value=httpx.Response(200, json=BODY)
    )
    amund.get_prompt("triage", label="staging")
    assert dict(route.calls.last.request.url.params) == {"label": "staging"}


@pytest.mark.req("SPF-01")
@respx.mock
def test_version_selector_has_no_label():
    route = respx.get(f"{BASE}/api/v1/prompts/triage").mock(
        return_value=httpx.Response(200, json=BODY)
    )
    amund.get_prompt("triage", version=3)
    assert dict(route.calls.last.request.url.params) == {"version": "3"}


@pytest.mark.req("SPF-01")
@respx.mock
def test_both_selectors_raise_before_any_request():
    route = respx.get(url__regex=r".*").mock(return_value=httpx.Response(200))
    with pytest.raises(ValueError):
        amund.get_prompt("triage", label="x", version=1)
    assert route.call_count == 0


@pytest.mark.req("SPF-01")
@respx.mock
def test_name_is_url_encoded():
    route = respx.get(url__regex=r".*").mock(
        return_value=httpx.Response(200, json=BODY)
    )
    amund.get_prompt("team/triage v2?")
    assert route.calls.last.request.url.raw_path.startswith(
        b"/api/v1/prompts/team%2Ftriage%20v2%3F?"
    )


@pytest.mark.req("SPF-02")
@respx.mock
def test_if_none_match_sent_and_304_is_not_modified():
    route = respx.get(f"{BASE}/api/v1/prompts/triage").mock(
        return_value=httpx.Response(304, headers={"ETag": '"abc"'})
    )
    p = amund.get_prompt("triage", if_none_match='"abc"')
    assert route.calls.last.request.headers["if-none-match"] == '"abc"'
    assert p.not_modified is True
    assert p.etag == '"abc"'
    assert p.version is None
    assert p.template_text is None
    assert p.labels is None


@pytest.mark.req("SPF-02")
@respx.mock
def test_no_if_none_match_header_by_default():
    route = respx.get(f"{BASE}/api/v1/prompts/triage").mock(
        return_value=httpx.Response(200, json=BODY)
    )
    amund.get_prompt("triage")
    assert "if-none-match" not in route.calls.last.request.headers


@pytest.mark.req("SPF-03")
@respx.mock
def test_403_raises_access_error_naming_scope():
    respx.get(f"{BASE}/api/v1/prompts/triage").mock(return_value=httpx.Response(403))
    with pytest.raises(PromptAccessError, match="prompts:read"):
        amund.get_prompt("triage")


@pytest.mark.req("SPF-03")
@respx.mock
def test_404_raises_not_found():
    respx.get(f"{BASE}/api/v1/prompts/nope").mock(return_value=httpx.Response(404))
    with pytest.raises(PromptNotFoundError):
        amund.get_prompt("nope")


@pytest.mark.req("SPF-03")
@respx.mock
def test_429_raises_rate_limit_with_retry_after_and_no_retry():
    route = respx.get(f"{BASE}/api/v1/prompts/triage").mock(
        return_value=httpx.Response(
            429,
            headers={"Retry-After": "7"},
            json={"error": {"code": "RATE_001", "message": "Too many"}},
        )
    )
    with pytest.raises(PromptRateLimitError) as exc:
        amund.get_prompt("triage")
    assert exc.value.retry_after == 7
    assert route.call_count == 1


@pytest.mark.req("SPF-03")
def test_errors_share_a_base_class():
    assert issubclass(PromptAccessError, PromptError)
    assert issubclass(PromptNotFoundError, PromptError)
    assert issubclass(PromptRateLimitError, PromptError)


@pytest.mark.req("SPF-03")
@respx.mock
def test_network_error_propagates_as_httpx_transport_error_without_retry():
    route = respx.get(f"{BASE}/api/v1/prompts/triage").mock(
        side_effect=httpx.ConnectError("refused")
    )
    with pytest.raises(httpx.TransportError):
        amund.get_prompt("triage")
    assert route.call_count == 1


@pytest.mark.req("SPF-03")
def test_not_initialized_raises_prompt_error(monkeypatch):
    monkeypatch.setattr(amund, "_client", None)
    monkeypatch.delenv("MORSE_API_KEY", raising=False)
    with pytest.raises(PromptError, match="AMUND_API_KEY"):
        amund.get_prompt("triage")


# --- Optional cache (SPF-04..06) ---------------------------------------------


class _Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


@pytest.fixture
def clock(monkeypatch):
    c = _Clock()
    monkeypatch.setattr(_prompts, "_clock", c)
    amund.clear_prompt_cache()
    amund.configure_prompt_cache(256)
    yield c
    amund.clear_prompt_cache()
    amund.configure_prompt_cache(256)


def _route():
    return respx.get(f"{BASE}/api/v1/prompts/triage")


@pytest.mark.req("SPF-04")
@respx.mock
def test_cache_off_by_default(clock):
    route = _route().mock(return_value=httpx.Response(200, json=BODY))
    amund.get_prompt("triage")
    amund.get_prompt("triage")
    amund.get_prompt("triage", cache_ttl=0)
    assert route.call_count == 3
    assert len(_prompts._cache) == 0


@pytest.mark.req("SPF-04")
@respx.mock
def test_cache_hit_within_ttl(clock):
    route = _route().mock(return_value=httpx.Response(200, json=BODY))
    a = amund.get_prompt("triage", cache_ttl=60)
    clock.t += 1
    b = amund.get_prompt("triage", cache_ttl=60)
    assert route.call_count == 1
    assert a == b


@pytest.mark.req("SPF-04")
@respx.mock
def test_caller_if_none_match_bypasses_cache(clock):
    route = _route().mock(return_value=httpx.Response(304, headers={"ETag": '"abc"'}))
    p = amund.get_prompt("triage", if_none_match='"abc"', cache_ttl=60)
    assert p.not_modified is True
    assert len(_prompts._cache) == 0
    assert route.call_count == 1


@pytest.mark.req("SPF-05")
@respx.mock
def test_stale_304_revalidates_and_refreshes(clock):
    route = _route().mock(
        side_effect=[
            httpx.Response(200, json=BODY),
            httpx.Response(304, headers={"ETag": '"abc"'}),
        ]
    )
    first = amund.get_prompt("triage", cache_ttl=60)
    clock.t += 61
    second = amund.get_prompt("triage", cache_ttl=60)
    assert route.calls.last.request.headers["if-none-match"] == '"abc"'
    assert second == first and second.not_modified is False
    clock.t += 30  # fresh again because the 304 refreshed the timestamp
    amund.get_prompt("triage", cache_ttl=60)
    assert route.call_count == 2


@pytest.mark.req("SPF-05")
@respx.mock
def test_stale_200_replaces_entry(clock):
    new = {**BODY, "version": 4, "etag": '"def"'}
    route = _route().mock(
        side_effect=[httpx.Response(200, json=BODY), httpx.Response(200, json=new)]
    )
    amund.get_prompt("triage", cache_ttl=60)
    clock.t += 61
    assert amund.get_prompt("triage", cache_ttl=60).version == 4
    assert amund.get_prompt("triage", cache_ttl=60).version == 4
    assert route.call_count == 2


@pytest.mark.req("SPF-05")
@pytest.mark.parametrize(
    ("status", "exc"), [(500, httpx.HTTPStatusError), (403, PromptAccessError)]
)
@respx.mock
def test_stale_entry_never_masks_errors(clock, status, exc):
    _route().mock(side_effect=[httpx.Response(200, json=BODY), httpx.Response(status)])
    amund.get_prompt("triage", cache_ttl=60)
    clock.t += 61
    with pytest.raises(exc):
        amund.get_prompt("triage", cache_ttl=60)


@pytest.mark.req("SPF-06")
@respx.mock
def test_lru_eviction_at_max_entries(clock):
    amund.configure_prompt_cache(max_entries=2)
    route = respx.get(url__regex=rf"{BASE}/api/v1/prompts/.*").mock(
        return_value=httpx.Response(200, json=BODY)
    )
    amund.get_prompt("a", cache_ttl=60)
    amund.get_prompt("b", cache_ttl=60)
    amund.get_prompt("a", cache_ttl=60)  # a is now most recent
    amund.get_prompt("c", cache_ttl=60)  # evicts b
    assert route.call_count == 3
    amund.get_prompt("a", cache_ttl=60)
    assert route.call_count == 3
    amund.get_prompt("b", cache_ttl=60)
    assert route.call_count == 4


@pytest.mark.req("SPF-06")
@respx.mock
def test_clear_prompt_cache(clock):
    route = _route().mock(return_value=httpx.Response(200, json=BODY))
    amund.get_prompt("triage", cache_ttl=60)
    amund.clear_prompt_cache()
    amund.get_prompt("triage", cache_ttl=60)
    assert route.call_count == 2


@pytest.mark.req("SPF-04", "SPF-06")
@respx.mock
def test_different_api_keys_do_not_share_entries(clock, monkeypatch):
    route = _route().mock(return_value=httpx.Response(200, json=BODY))
    amund.get_prompt("triage", cache_ttl=60)
    monkeypatch.setattr(
        amund._client, "_api_key", "mk_test_other"
    )  # pragma: allowlist secret
    amund.get_prompt("triage", cache_ttl=60)
    assert route.call_count == 2


@pytest.mark.req("SPF-04")
@respx.mock
def test_label_and_version_are_separate_entries(clock):
    route = _route().mock(return_value=httpx.Response(200, json=BODY))
    amund.get_prompt("triage", cache_ttl=60)
    amund.get_prompt("triage", version=3, cache_ttl=60)
    assert route.call_count == 2


@pytest.mark.req("SPF-06")
@respx.mock
def test_cache_thread_safety_smoke(clock):
    amund.configure_prompt_cache(max_entries=4)
    _route().mock(return_value=httpx.Response(200, json=BODY))
    errors: list[BaseException] = []

    def work() -> None:
        try:
            for _ in range(50):
                amund.get_prompt("triage", cache_ttl=60)
                amund.clear_prompt_cache()
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=work) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
