"""AWS Strands Agents adapter — route Strands' own OpenTelemetry spans to Amund.

MHQ-133. Strands traces itself: ``strands.telemetry.tracer.Tracer`` emits
``invoke_agent`` / ``execute_event_loop_cycle`` / ``chat`` / ``execute_tool``
(and ``invoke_swarm`` / ``invoke_graph``) spans with GenAI semantic-convention
attributes, through whatever OpenTelemetry ``TracerProvider`` is global. The
Amund API already maps those conventions to typed spans, tokens and cost on
its OTLP endpoint, so this adapter does not rebuild Strands' spans as
``SpanData``: it only decides where Strands' spans go.

Routing, decided when each span starts:

- **No OTel SDK provider is configured** (the zero-code case: the global
  provider is OTel's no-op proxy, so Strands' spans would be dropped). Strands'
  spans are started on the Amund SDK's own provider (``otel_provider``), so they
  are sampled, redacted and exported exactly like ``run()`` / ``span()``
  traces. Nothing global is changed: other OTel instrumentation in the process
  is not captured.
- **The app configured its own SDK provider** (``StrandsTelemetry()``, an OTLP
  exporter to another backend, ...). Strands' spans stay on that provider and
  reach its exporters unchanged; a processor added to it forwards copies of the
  spans from Strands' instrumentation scope to Amund.

Also:

- While Strands invokes a model it marks itself the LLM owner
  (``set_llm_owner``), so the zero-code OpenAI / google-genai patches do not
  record the same call a second time (Strands' ``OpenAIModel``,
  ``GeminiModel``, LiteLLM's OpenAI provider).
- Multi-agent spans (``invoke_swarm`` / ``invoke_graph``) are stamped
  ``morse.span_type=agent``. Without it the server types any span carrying a
  ``gen_ai.*`` attribute as ``llm``, which would show the swarm as an LLM call
  with no model.
- A Strands trace is separate from any open ``amund.run()`` trace (the Amund
  SDK does not set the OTel current span), so wrapping a Strands agent in
  ``run()`` yields two traces.
"""

from __future__ import annotations

import importlib.metadata
import logging
import threading
from typing import Any

from amund._internal.active_framework import reset_llm_owner, set_llm_owner
from amund._internal.errors import absorb_errors

logger = logging.getLogger("amund")

_FRAMEWORK_NAME = "strands"
_SCOPE_PREFIX = "strands"
_INSTALL_FLAG = "_morse_ai_installed"
_MULTIAGENT_OPERATIONS = ("invoke_swarm", "invoke_graph")

# (owner object, attribute, original) for uninstall().
_patched: list[tuple[Any, str, Any]] = []
_owner_tokens: dict[int, Any] = {}
_owner_lock = threading.Lock()
_forwarded_providers: set[int] = set()
_forward_lock = threading.Lock()


def _strands_version() -> str:
    try:
        return importlib.metadata.version("strands-agents")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


# ---------------------------------------------------------------------------
# Customer-owned provider: forward Strands-scope spans to Amund
# ---------------------------------------------------------------------------


def _copy_span(span: Any) -> Any:
    """A detached ``ReadableSpan`` copy, so redacting it for Amund never
    changes what the app's own exporters receive."""
    from opentelemetry.sdk.trace import ReadableSpan  # noqa: PLC0415

    return ReadableSpan(
        name=span.name,
        context=span.context,
        parent=span.parent,
        resource=span.resource,
        attributes=dict(span.attributes or {}),
        events=tuple(span.events or ()),
        links=tuple(span.links or ()),
        kind=span.kind,
        status=span.status,
        start_time=span.start_time,
        end_time=span.end_time,
        instrumentation_scope=span.instrumentation_scope,
    )


def _make_forwarder() -> Any:
    from opentelemetry.sdk.trace import SpanProcessor  # noqa: PLC0415
    from opentelemetry.sdk.trace.export import (  # noqa: PLC0415
        BatchSpanProcessor,
        SpanExporter,
        SpanExportResult,
    )

    from amund.otel_provider import get_engine  # noqa: PLC0415

    class _ForwardToMorseExporter(SpanExporter):
        def export(self, spans: Any) -> Any:
            exporter = get_engine().exporter
            if exporter is None:  # amund.init() not called (or shut down)
                return SpanExportResult.SUCCESS
            try:
                return exporter.export([_copy_span(s) for s in spans])
            except Exception as exc:  # noqa: BLE001 — never break the app
                logger.debug("amund: strands span forward failed: %s", exc)
                return SpanExportResult.FAILURE

        def shutdown(self) -> None:
            return None

    class _StrandsScopeProcessor(SpanProcessor):
        def __init__(self) -> None:
            self._batch = BatchSpanProcessor(_ForwardToMorseExporter())

        def on_start(self, span: Any, parent_context: Any = None) -> None:
            return None

        def on_end(self, span: Any) -> None:
            scope = getattr(span, "instrumentation_scope", None)
            name = getattr(scope, "name", "") or ""
            if name.startswith(_SCOPE_PREFIX):
                self._batch.on_end(span)

        def shutdown(self) -> None:
            self._batch.shutdown()

        def force_flush(self, timeout_millis: int = 30000) -> bool:
            return self._batch.force_flush(timeout_millis)

    return _StrandsScopeProcessor()


def _ensure_forwarder(provider: Any) -> None:
    key = id(provider)
    if key in _forwarded_providers:
        return
    with _forward_lock:
        if key in _forwarded_providers:
            return
        provider.add_span_processor(_make_forwarder())
        _forwarded_providers.add(key)
        logger.info(
            "amund: Strands spans are exported by the app's TracerProvider; "
            "forwarding a copy to Amund"
        )


# ---------------------------------------------------------------------------
# The tracer Strands uses
# ---------------------------------------------------------------------------


class _RoutingTracer:
    """Stands in for ``Tracer.tracer``; picks the provider per span."""

    def __init__(self, inner: Any) -> None:
        self._av_inner = inner
        self._av_cache: tuple[int, Any] | None = None

    def _av_target(self) -> Any:
        from opentelemetry import trace as trace_api  # noqa: PLC0415
        from opentelemetry.sdk.trace import TracerProvider  # noqa: PLC0415

        from amund.otel_provider import get_engine  # noqa: PLC0415

        global_provider = trace_api.get_tracer_provider()
        if isinstance(global_provider, TracerProvider):
            _ensure_forwarder(global_provider)
            return self._av_inner
        provider = get_engine().provider
        if provider is None:
            return self._av_inner  # no init(): Strands' default (no-op) behavior
        cache = self._av_cache
        if cache is None or cache[0] != id(provider):
            tracer = provider.get_tracer("strands.telemetry.tracer", _strands_version())
            cache = (id(provider), tracer)
            self._av_cache = cache
        return cache[1]

    def start_span(self, *args: Any, **kwargs: Any) -> Any:
        return self._av_target().start_span(*args, **kwargs)

    def start_as_current_span(self, *args: Any, **kwargs: Any) -> Any:
        return self._av_target().start_as_current_span(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._av_inner, name)


def _route(tracer_obj: Any) -> None:
    inner = getattr(tracer_obj, "tracer", None)
    if inner is not None and not isinstance(inner, _RoutingTracer):
        tracer_obj.tracer = _RoutingTracer(inner)


# ---------------------------------------------------------------------------
# Method patches
# ---------------------------------------------------------------------------


def _wrap_init(original: Any) -> Any:
    def __init__(self: Any, *args: Any, **kwargs: Any) -> None:  # noqa: N807
        original(self, *args, **kwargs)
        try:
            _route(self)
        except Exception as exc:  # noqa: BLE001
            logger.debug("amund: strands tracer routing failed: %s", exc)

    __init__.__wrapped__ = original  # type: ignore[attr-defined]
    return __init__


def _wrap_start_model(original: Any) -> Any:
    def start_model_invoke_span(self: Any, *args: Any, **kwargs: Any) -> Any:
        span = original(self, *args, **kwargs)
        try:
            token = set_llm_owner(_FRAMEWORK_NAME)
            with _owner_lock:
                _owner_tokens[id(span)] = token
        except Exception as exc:  # noqa: BLE001
            logger.debug("amund: strands llm-owner mark failed: %s", exc)
        return span

    start_model_invoke_span.__wrapped__ = original  # type: ignore[attr-defined]
    return start_model_invoke_span


def _wrap_end(original: Any) -> Any:
    def end(self: Any, span: Any, *args: Any, **kwargs: Any) -> Any:
        with _owner_lock:
            token = _owner_tokens.pop(id(span), None)
        if token is not None:
            reset_llm_owner(token)
        return original(self, span, *args, **kwargs)

    end.__wrapped__ = original  # type: ignore[attr-defined]
    return end


def _wrap_start_multiagent(original: Any) -> Any:
    def start_multiagent_span(self: Any, *args: Any, **kwargs: Any) -> Any:
        span = original(self, *args, **kwargs)
        try:
            span.set_attribute("morse.span_type", "agent")
        except Exception as exc:  # noqa: BLE001
            logger.debug("amund: strands multi-agent span typing failed: %s", exc)
        return span

    start_multiagent_span.__wrapped__ = original  # type: ignore[attr-defined]
    return start_multiagent_span


_METHOD_WRAPPERS = (
    ("__init__", _wrap_init),
    ("start_model_invoke_span", _wrap_start_model),
    ("end_model_invoke_span", _wrap_end),
    ("end_span_with_error", _wrap_end),
    ("end_span_with_cancellation", _wrap_end),
    ("start_multiagent_span", _wrap_start_multiagent),
)


def is_installed() -> bool:
    try:
        import strands  # type: ignore[import-not-found]
    except ImportError:
        return False
    return bool(getattr(strands, _INSTALL_FLAG, False))


@absorb_errors(fingerprint="morse_ai.adapters.strands.install")
def install() -> bool:
    """Route Strands Agents' OTel spans to Amund. Idempotent.

    Returns True if installed (newly or already), False if Strands is not
    available or its tracer has an unexpected shape.
    """
    try:
        import strands  # type: ignore[import-not-found]
        from strands.telemetry import (
            tracer as st_tracer,  # type: ignore[import-not-found]
        )
    except ImportError:
        return False

    if getattr(strands, _INSTALL_FLAG, False):
        return True

    tracer_cls = getattr(st_tracer, "Tracer", None)
    if tracer_cls is None or "__init__" not in tracer_cls.__dict__:
        logger.warning(
            "amund: strands.telemetry.tracer.Tracer not found; Strands "
            "spans will not reach Amund"
        )
        return False

    for attr, wrapper in _METHOD_WRAPPERS:
        original = tracer_cls.__dict__.get(attr)
        if original is None:
            logger.debug("amund: strands Tracer.%s missing; not patched", attr)
            continue
        setattr(tracer_cls, attr, wrapper(original))
        _patched.append((tracer_cls, attr, original))

    # Agents created before install() already hold the singleton.
    existing = getattr(st_tracer, "_tracer_instance", None)
    if existing is not None:
        _route(existing)

    strands._morse_ai_installed = True  # type: ignore[attr-defined]
    logger.info("amund: Strands adapter installed")
    return True


@absorb_errors(fingerprint="morse_ai.adapters.strands.uninstall")
def uninstall() -> None:
    """Restore Strands' tracer. Idempotent. Forwarders already added to an
    app-owned provider stay (they no-op once Amund is shut down)."""
    while _patched:
        owner, attr, original = _patched.pop()
        setattr(owner, attr, original)
    try:
        import strands  # type: ignore[import-not-found]
        from strands.telemetry import (
            tracer as st_tracer,  # type: ignore[import-not-found]
        )
    except ImportError:
        return
    existing = getattr(st_tracer, "_tracer_instance", None)
    inner = getattr(existing, "tracer", None)
    if isinstance(inner, _RoutingTracer):
        existing.tracer = inner._av_inner
    strands._morse_ai_installed = False  # type: ignore[attr-defined]
