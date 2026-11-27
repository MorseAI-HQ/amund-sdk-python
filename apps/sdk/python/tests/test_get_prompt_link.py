"""UP-09: get_prompt links the open run to the fetched prompt version.

HTTP is mocked with respx; nothing touches the network.
"""

from __future__ import annotations

import logging
import threading
from types import SimpleNamespace

import httpx
import pytest
import respx

import amund
from amund import otel_provider, tracing

BASE = "https://api.morse.test"
BODY = {
    "name": "triage",
    "version": 3,
    "template_text": "Hello {{user}}",
    "labels": ["production"],
    "etag": '"abc"',
}
FETCH_REF = {"name": "triage", "version": 3, "link_source": "fetch"}


@pytest.fixture(autouse=True)
def linkenv(monkeypatch):
    amund._reset_for_testing()
    amund.clear_prompt_cache()
    tracing._transport = None
    tracing._last_trace = None
    monkeypatch.setattr(
        amund,
        "_client",
        SimpleNamespace(
            endpoint=f"{BASE}/api/v1/traces/batch",
            _api_key="mk_test_fake",  # pragma: allowlist secret
        ),
    )
    yield
    amund.clear_prompt_cache()
    monkeypatch.setattr(amund, "_client", None)  # fake has no reset hook
    amund._reset_for_testing()


def _route():
    return respx.get(f"{BASE}/api/v1/prompts/triage")


def _prompts_of_last_trace():
    payload = tracing._get_last_trace()
    return (payload or {}).get("trace", {}).get("prompts")


@pytest.mark.req("UP-09")
@respx.mock
def test_fetch_inside_run_links_with_link_source():
    _route().mock(return_value=httpx.Response(200, json=BODY))
    with amund.run("agent"):
        amund.get_prompt("triage")
    assert _prompts_of_last_trace() == [FETCH_REF]


@pytest.mark.req("UP-09")
@respx.mock
def test_fetch_inside_span_pins_span_id():
    _route().mock(return_value=httpx.Response(200, json=BODY))
    with amund.run("agent"), amund.span("llm"):
        amund.get_prompt("triage")
    ref = _prompts_of_last_trace()[0]
    assert ref["link_source"] == "fetch" and ref["span_id"]


@pytest.mark.req("UP-09")
@respx.mock
def test_no_active_run_is_noop():
    _route().mock(return_value=httpx.Response(200, json=BODY))
    assert amund.get_prompt("triage").version == 3
    assert tracing.get_active_trace() is None


@pytest.mark.req("UP-09")
@respx.mock
def test_link_false_never_links():
    _route().mock(return_value=httpx.Response(200, json=BODY))
    with amund.run("agent"):
        amund.get_prompt("triage", link=False)
    assert not _prompts_of_last_trace()


@pytest.mark.req("UP-09")
@respx.mock
def test_cache_hit_links_a_new_run():
    route = _route().mock(return_value=httpx.Response(200, json=BODY))
    amund.get_prompt("triage", cache_ttl=60)  # warm cache, no run
    with amund.run("agent"):
        amund.get_prompt("triage", cache_ttl=60)
    assert route.call_count == 1
    assert _prompts_of_last_trace() == [FETCH_REF]


@pytest.mark.req("UP-09")
@respx.mock
def test_cache_hit_in_thread_without_run_does_not_link():
    _route().mock(return_value=httpx.Response(200, json=BODY))
    amund.get_prompt("triage", cache_ttl=60)
    seen = {}

    def worker():
        seen["prompt"] = amund.get_prompt("triage", cache_ttl=60)

    with amund.run("agent"):
        t = threading.Thread(target=worker)  # fresh thread: empty contextvars
        t.start()
        t.join()
    assert seen["prompt"].version == 3
    assert not _prompts_of_last_trace()


@pytest.mark.req("UP-09")
@respx.mock
def test_304_links_via_cached_prompt_only():
    _route().mock(
        side_effect=[
            httpx.Response(200, json=BODY),
            httpx.Response(304, headers={"ETag": '"abc"'}),
            httpx.Response(304, headers={"ETag": '"abc"'}),
        ]
    )
    amund.get_prompt("triage", cache_ttl=60)  # cache populated
    with amund.run("agent"):
        p = amund.get_prompt("triage", if_none_match='"abc"')
    assert p.not_modified is True
    assert _prompts_of_last_trace() == [FETCH_REF]
    amund.clear_prompt_cache()
    with amund.run("agent2"):
        assert amund.get_prompt("triage", if_none_match='"abc"').not_modified
    assert not _prompts_of_last_trace()


@pytest.mark.req("UP-09")
@respx.mock
def test_dedupe_with_set_prompt_prefers_fetch():
    _route().mock(return_value=httpx.Response(200, json=BODY))
    with amund.run("agent"):
        amund.set_prompt("triage", 3)
        amund.get_prompt("triage")
    assert _prompts_of_last_trace() == [FETCH_REF]
    with amund.run("agent"):
        amund.get_prompt("triage")
        amund.set_prompt("triage", 3)
    assert _prompts_of_last_trace() == [FETCH_REF]


@pytest.mark.req("UP-09")
def test_set_prompt_ref_omits_link_source():
    with amund.run("agent"):
        amund.set_prompt("triage", 3)
    assert _prompts_of_last_trace() == [{"name": "triage", "version": 3}]


@pytest.mark.req("UP-09")
@respx.mock
def test_linking_never_raises(monkeypatch):
    _route().mock(return_value=httpx.Response(200, json=BODY))

    def boom(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(amund, "link_fetched_prompt", boom)
    with amund.run("agent"):
        assert amund.get_prompt("triage").version == 3


@pytest.mark.req("UP-09")
@respx.mock
def test_one_info_log_per_process(caplog):
    _route().mock(return_value=httpx.Response(200, json=BODY))
    with caplog.at_level(logging.INFO, logger="amund"):
        with amund.run("a"):
            amund.get_prompt("triage")
        with amund.run("b"):
            amund.get_prompt("triage")
    msgs = [r.getMessage() for r in caplog.records if "link=False" in r.getMessage()]
    assert len(msgs) == 1


@pytest.mark.req("UP-09")
def test_otlp_attribute_link_source():
    def attrs(ref):
        return otel_provider._span_dict_to_otel_attributes(
            {"type": "agent"},
            {},
            {},
            is_root=True,
            max_field_bytes=1000,
            prompt_ref=ref,
        )

    assert attrs(FETCH_REF)["amund.prompt.link_source"] == "fetch"
    assert "amund.prompt.link_source" not in attrs({"name": "triage", "version": 3})
