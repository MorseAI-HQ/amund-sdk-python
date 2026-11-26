"""Google Gen AI SDK adapter (``google-genai``) — MHQ-1506.

Traces ``client.models.generate_content`` / ``generate_content_stream`` and the
async ``client.aio.models`` variants, for both the Gemini Developer API and
Vertex AI (``genai.Client(vertexai=True)``). Zero-code: the auto-detect layer
calls ``install()`` when ``google.genai`` is importable, so an app that only
sets ``AMUND_API_KEY`` and calls ``amund.init()`` is traced.

What is patched, and why the private methods
---------------------------------------------
The public ``generate_content`` runs a loop: automatic function calling (AFC)
calls the model, runs the Python tools the caller passed, and calls the model
again; continuation tokens add further requests. Each of those requests goes
through ``Models._generate_content`` (``_generate_content_stream`` for
streams), and each returns its own ``usage_metadata``. Patching the private
per-request methods therefore yields one ``llm`` span per billed request, so
cost is right for AFC and continuation calls. The public result carries only
the last request's usage, so patching it would undercount. If a future SDK
drops the private methods, ``install()`` falls back to the public ones (one
span per call, documented as an undercount for AFC).

Usage mapping (``usage_metadata``)
----------------------------------
- ``input_tokens``  = ``prompt_token_count`` + ``tool_use_prompt_token_count``
- ``output_tokens`` = ``candidates_token_count`` + ``thoughts_token_count``
  (Gemini bills thinking tokens at the output rate; ``candidates`` excludes them)
- ``reasoning_tokens`` = ``thoughts_token_count``
- ``cached_input_tokens`` = ``cached_content_token_count`` (included in
  ``prompt_token_count``, so pricing subtracts it from the uncached input)

Provider is recorded as ``metadata.provider`` (``gemini`` or ``vertex_ai``),
with ``metadata.location`` for Vertex.

Framework double-recording: a call made while a framework adapter owns the
model call (``get_llm_owner()``) or inside an open framework ``llm`` span (for
example LangChain ``ChatGoogleGenerativeAI`` under ``MorseCallbackHandler``) is
not recorded again. Google ADK has no Amund adapter today, so ADK's model calls
are recorded here; ADK's own OpenTelemetry spans are not exported to Amund.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import UTC, datetime
from typing import Any

from amund._internal.active_framework import (
    clear_active_framework,
    get_llm_owner,
    set_active_framework,
)
from amund._internal.errors import absorb_errors
from amund.context.decomposer import decompose_openai_messages
from amund.pricing import calculate_cost
from amund.spans import SpanData, SpanType
from amund.tracing import _finalize_span, get_active_trace
from amund.tracing import run as _av_run

logger = logging.getLogger("amund")

_FRAMEWORK_NAME = "google-genai"
_INSTALL_FLAG = "_morse_ai_installed"
_CONTENT_BYTE_CAP = 4096

_SPAN_NAME = "google_genai.models.generate_content"
_STREAM_SPAN_NAME = "google_genai.models.generate_content_stream"

# (class name, private per-request method, public fallback, is_async, is_stream)
_TARGETS: tuple[tuple[str, str, str, bool, bool], ...] = (
    ("Models", "_generate_content", "generate_content", False, False),
    ("Models", "_generate_content_stream", "generate_content_stream", False, True),
    ("AsyncModels", "_generate_content", "generate_content", True, False),
    (
        "AsyncModels",
        "_generate_content_stream",
        "generate_content_stream",
        True,
        True,
    ),
)

# (class, attribute name, original function) for uninstall().
_patched: list[tuple[Any, str, Any]] = []


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _field(obj: Any, name: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _truncate(text: str) -> str:
    encoded = text.encode("utf-8", errors="replace")
    if len(encoded) <= _CONTENT_BYTE_CAP:
        return text
    return encoded[:_CONTENT_BYTE_CAP].decode("utf-8", errors="ignore")


def _json(value: Any) -> str:
    try:
        return json.dumps(value, default=str, sort_keys=True)
    except (TypeError, ValueError):
        return str(value)


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _normalize_model(model: Any) -> str | None:
    """``models/gemini-x`` / ``projects/../publishers/google/models/gemini-x``
    / ``google/gemini-x`` -> ``gemini-x``. Tuned-model endpoints stay as-is."""
    if not isinstance(model, str) or not model:
        return None
    if "/models/" in model:
        return model.rsplit("/models/", 1)[1]
    for prefix in ("models/", "publishers/google/models/", "google/"):
        if model.startswith(prefix):
            return model[len(prefix) :]
    return model


def _provider_facts(models_obj: Any) -> dict[str, Any]:
    api_client = getattr(models_obj, "_api_client", None)
    vertex = bool(getattr(api_client, "vertexai", False))
    facts: dict[str, Any] = {"provider": "vertex_ai" if vertex else "gemini"}
    location = getattr(api_client, "location", None) if vertex else None
    if isinstance(location, str) and location:
        facts["location"] = location
    return facts


# ---------------------------------------------------------------------------
# Request / response reading
# ---------------------------------------------------------------------------

_ROLE_MAP = {"model": "assistant", "function": "tool", "tool": "tool"}


def _part_text(part: Any) -> str | None:
    if isinstance(part, str):
        return part
    if _field(part, "thought"):
        return None  # thought summaries are not the model's answer
    text = _field(part, "text")
    if isinstance(text, str) and text:
        return text
    call = _field(part, "function_call")
    if call is not None:
        args = _field(call, "args")
        name = _field(call, "name") or "tool"
        return f"[tool_call:{name}] {_json(args)}" if args else f"[tool_call:{name}]"
    result = _field(part, "function_response")
    if result is not None:
        name = _field(result, "name") or "tool"
        return f"[tool_result:{name}] {_json(_field(result, 'response'))}"
    inline = _field(part, "inline_data")
    if inline is not None:
        return f"[inline_data:{_field(inline, 'mime_type') or 'unknown'}]"
    file_data = _field(part, "file_data")
    if file_data is not None:
        return f"[file:{_field(file_data, 'file_uri') or ''}]"
    return None


def _content_items(contents: Any) -> list[Any]:
    if contents is None:
        return []
    if isinstance(contents, (str, dict)) or not isinstance(contents, (list, tuple)):
        return [contents]
    return list(contents)


def _to_message(item: Any) -> dict[str, Any] | None:
    """One ``contents`` element -> ``{"role", "content"}`` (OpenAI shape)."""
    parts = _field(item, "parts") if not isinstance(item, str) else None
    if parts is None:
        # A bare string or Part is a user turn.
        text = _part_text(item)
        return {"role": "user", "content": _truncate(text)} if text else None
    role = _field(item, "role") or "user"
    texts = [t for t in (_part_text(p) for p in parts or []) if t]
    if not texts:
        return None
    return {"role": _ROLE_MAP.get(role, role), "content": _truncate("\n".join(texts))}


def _system_text(config: Any) -> str | None:
    instruction = _field(config, "system_instruction")
    if instruction is None:
        return None
    if isinstance(instruction, str):
        return instruction
    parts = _field(instruction, "parts")
    items = parts if parts is not None else _content_items(instruction)
    texts = [t for t in (_part_text(p) for p in items) if t]
    return "\n".join(texts) or None


def _request_messages(kwargs: dict[str, Any]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    try:
        system = _system_text(kwargs.get("config"))
        if system:
            messages.append({"role": "system", "content": _truncate(system)})
        for item in _content_items(kwargs.get("contents")):
            msg = _to_message(item)
            if msg:
                messages.append(msg)
    except Exception as exc:  # noqa: BLE001
        logger.debug("amund: google-genai input capture swallowed: %s", exc)
    return messages


def _request_tools(kwargs: dict[str, Any]) -> list[dict[str, Any]] | None:
    """Function declarations as OpenAI-shaped tool dicts (for segments only)."""
    tools: list[dict[str, Any]] = []
    for tool in _field(kwargs.get("config"), "tools") or []:
        if callable(tool) and not hasattr(tool, "function_declarations"):
            tools.append({"name": getattr(tool, "__name__", "tool")})
            continue
        for decl in _field(tool, "function_declarations") or []:
            tools.append(
                {
                    "name": _field(decl, "name") or "tool",
                    "description": _field(decl, "description") or "",
                }
            )
    return tools or None


def _function_responses(kwargs: dict[str, Any]) -> list[tuple[str | None, str, Any]]:
    """``(id, name, response)`` for every tool result carried by a request."""
    out: list[tuple[str | None, str, Any]] = []
    for item in _content_items(kwargs.get("contents")):
        parts = _field(item, "parts")
        for part in parts if parts is not None else [item]:
            result = _field(part, "function_response") if part is not None else None
            if result is not None:
                out.append(
                    (
                        _field(result, "id"),
                        _field(result, "name") or "tool",
                        _field(result, "response"),
                    )
                )
    return out


def _tool_key(call_id: Any, name: str) -> str:
    # Gemini API function calls usually carry no id: match on name instead.
    return f"genai:{call_id}" if call_id else f"genai-name:{name}"


class _Reply:
    """What a response (or a whole stream) said, in one normalized shape."""

    def __init__(self) -> None:
        self.model_version: str | None = None
        self.response_id: str | None = None
        self.usage: Any = None
        self.texts: list[str] = []
        self.function_calls: list[tuple[Any, str, Any]] = []
        self.finish_reason: str | None = None
        self.chunks = 0

    def feed(self, response: Any) -> None:
        self.chunks += 1
        version = _field(response, "model_version")
        if isinstance(version, str) and version:
            self.model_version = version
        rid = _field(response, "response_id")
        if isinstance(rid, str) and rid:
            self.response_id = rid
        usage = _field(response, "usage_metadata")
        if usage is not None:
            self.usage = usage  # streamed usage is cumulative; the last one wins
        candidates = _field(response, "candidates") or []
        if not candidates:
            return
        candidate = candidates[0]
        reason = _field(candidate, "finish_reason")
        if reason is not None:
            self.finish_reason = str(getattr(reason, "value", reason))
        for part in _field(_field(candidate, "content"), "parts") or []:
            call = _field(part, "function_call")
            if call is not None:
                self.function_calls.append(
                    (
                        _field(call, "id"),
                        _field(call, "name") or "tool",
                        _field(call, "args"),
                    )
                )
            text = _part_text(part)
            if text:
                self.texts.append(text)

    def output_text(self, streamed: bool) -> str | None:
        if not self.texts:
            return None
        joined = "".join(self.texts) if streamed else "\n".join(self.texts)
        return _truncate(joined)


def extract_usage(usage: Any, model: str | None) -> dict[str, Any]:
    """Token counts and catalog cost from a ``usage_metadata`` object."""
    if usage is None:
        return {}
    prompt = _int(_field(usage, "prompt_token_count"))
    tool_use = _int(_field(usage, "tool_use_prompt_token_count"))
    candidates = _int(_field(usage, "candidates_token_count"))
    thoughts = _int(_field(usage, "thoughts_token_count"))
    cached = _int(_field(usage, "cached_content_token_count"))
    input_tokens = prompt + tool_use
    output_tokens = candidates + thoughts
    total = _int(_field(usage, "total_token_count")) or input_tokens + output_tokens
    result: dict[str, Any] = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total,
        "cached_input_tokens": cached,
    }
    if thoughts:
        result["reasoning_tokens"] = thoughts
    if tool_use:
        result["tool_use_prompt_tokens"] = tool_use
    if model:
        cost = calculate_cost(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=model,
            cached_input_tokens=cached,
        )
        if cost is not None:
            result["cost_usd"] = round(cost, 6)
    return result


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------


def _framework_records_call() -> bool:
    """True when a framework adapter already records this model call."""
    if get_llm_owner() is not None:
        return True
    state = get_active_trace()
    return bool(state is not None and state.span_stack) and (
        state.span_stack[-1].type == SpanType.LLM  # type: ignore[union-attr]
    )


def _record_tool_spans_for_request(kwargs: dict[str, Any]) -> None:
    state = get_active_trace()
    if state is None or not state.pending_tool_calls:
        return
    try:
        for call_id, name, output in _function_responses(kwargs):
            pending = state.pending_tool_calls.pop(_tool_key(call_id, name), None)
            if pending is None:
                continue
            metadata: dict[str, Any] = {
                "timing": "inferred",
                "source": "google_genai.function_call",
            }
            if call_id:
                metadata["tool_call_id"] = call_id
            tool_span = SpanData(
                trace_id=state.trace_id,
                type=SpanType.TOOL,
                name=name,
                parent_span_id=(
                    state.span_stack[-1].span_id if state.span_stack else None
                ),
                input_data=(
                    {"arguments": pending["arguments"]}
                    if pending["arguments"]
                    else None
                ),
                output_data={"content": _truncate(_json(output))},
                metadata=metadata,
            )
            tool_span.started_at = pending["asked_iso"]
            _finalize_span(tool_span, pending["asked_mono"])
            state.completed_spans.append(tool_span)
    except Exception as exc:  # noqa: BLE001
        logger.debug("amund: google-genai tool span recording swallowed: %s", exc)


class _CallState:
    def __init__(self, models_obj: Any, kwargs: dict[str, Any], stream: bool) -> None:
        self.kwargs = kwargs
        self.stream = stream
        self.name = _STREAM_SPAN_NAME if stream else _SPAN_NAME
        self.provider = _provider_facts(models_obj)
        self.start_mono = time.monotonic()
        self.started_at_iso = datetime.now(UTC).isoformat()
        self.first_chunk_mono: float | None = None
        self.done = False

    def restart_clock(self) -> None:
        """Start the llm span after the inferred tool spans just closed (MHQ-1921)."""
        self.start_mono = time.monotonic()
        self.started_at_iso = datetime.now(UTC).isoformat()

    def record(self, reply: _Reply | None, error: BaseException | None = None) -> None:
        if self.done:
            return
        self.done = True
        state = get_active_trace()
        if state is None:
            return
        try:
            reply = reply or _Reply()
            model = reply.model_version or _normalize_model(self.kwargs.get("model"))
            telemetry = extract_usage(reply.usage, model)
            messages = _request_messages(self.kwargs)
            segments = decompose_openai_messages(messages, _request_tools(self.kwargs))
            output_text = reply.output_text(self.stream)
            metadata: dict[str, Any] = dict(self.provider)
            if reply.finish_reason:
                metadata["finish_reason"] = reply.finish_reason
            if reply.response_id:
                metadata["response_id"] = reply.response_id
            if telemetry.get("tool_use_prompt_tokens"):
                metadata["tool_use_prompt_tokens"] = telemetry["tool_use_prompt_tokens"]
            if self.stream:
                metadata["streamed"] = True
                if self.first_chunk_mono is not None:
                    metadata["time_to_first_chunk_ms"] = int(
                        (self.first_chunk_mono - self.start_mono) * 1000
                    )
            if reply.usage is None and error is None:
                metadata["usage_unavailable"] = True
            span = SpanData(
                trace_id=state.trace_id,
                type=SpanType.LLM,
                name=self.name,
                parent_span_id=(
                    state.span_stack[-1].span_id if state.span_stack else None
                ),
                model=model,
                input_tokens=telemetry.get("input_tokens"),
                output_tokens=telemetry.get("output_tokens"),
                total_tokens=telemetry.get("total_tokens"),
                cost_usd=telemetry.get("cost_usd"),
                cached_input_tokens=telemetry.get("cached_input_tokens"),
                reasoning_tokens=telemetry.get("reasoning_tokens"),
                context_segments=segments or None,
                input_data={"messages": messages} if messages else None,
                output_data={"content": output_text} if output_text else None,
                metadata=metadata,
            )
            span.started_at = self.started_at_iso
            _finalize_span(
                span,
                self.start_mono,
                status="failed" if error is not None else "completed",
                error_message=f"{type(error).__name__}: {error}" if error else None,
                exc=error,
            )
            state.completed_spans.append(span)
            if error is None and reply.function_calls:
                asked_mono = time.monotonic()
                asked_iso = datetime.now(UTC).isoformat()
                for call_id, name, args in reply.function_calls:
                    state.pending_tool_calls[_tool_key(call_id, name)] = {
                        "name": name,
                        "arguments": args,
                        "asked_mono": asked_mono,
                        "asked_iso": asked_iso,
                        "llm_span_id": span.span_id,
                    }
        except Exception as exc:  # noqa: BLE001
            logger.debug("amund: google-genai span recording swallowed: %s", exc)

    def record_in_trace(
        self, reply: _Reply | None, error: BaseException | None = None
    ) -> None:
        """Record from a stream's end, opening a trace if none is active."""
        set_active_framework(_FRAMEWORK_NAME)
        try:
            if get_active_trace() is not None:
                self.record(reply, error)
            else:
                with _av_run(self.name):
                    self.record(reply, error)
        except Exception as exc:  # noqa: BLE001 — never break the caller's stream
            logger.debug("amund: google-genai stream recording failed: %s", exc)
        finally:
            clear_active_framework()


def _reply_of(response: Any) -> _Reply:
    reply = _Reply()
    if response is not None:
        reply.feed(response)
    return reply


def _traced_sync(original: Any) -> Any:
    def traced(self: Any, *args: Any, **kwargs: Any) -> Any:
        if _framework_records_call():
            return original(self, *args, **kwargs)
        call_state = _CallState(self, kwargs, stream=False)

        def _run() -> Any:
            _record_tool_spans_for_request(kwargs)
            call_state.restart_clock()
            try:
                response = original(self, *args, **kwargs)
            except BaseException as exc:
                call_state.record(None, exc)
                raise
            call_state.record(_reply_of(response))
            return response

        set_active_framework(_FRAMEWORK_NAME)
        try:
            if get_active_trace() is not None:
                return _run()
            with _av_run(call_state.name):
                return _run()
        finally:
            clear_active_framework()

    traced.__wrapped__ = original  # type: ignore[attr-defined]
    return traced


def _traced_async(original: Any) -> Any:
    async def traced(self: Any, *args: Any, **kwargs: Any) -> Any:
        if _framework_records_call():
            return await original(self, *args, **kwargs)
        call_state = _CallState(self, kwargs, stream=False)

        async def _run() -> Any:
            _record_tool_spans_for_request(kwargs)
            call_state.restart_clock()
            try:
                response = await original(self, *args, **kwargs)
            except BaseException as exc:
                call_state.record(None, exc)
                raise
            call_state.record(_reply_of(response))
            return response

        set_active_framework(_FRAMEWORK_NAME)
        try:
            if get_active_trace() is not None:
                return await _run()
            with _av_run(call_state.name):
                return await _run()
        finally:
            clear_active_framework()

    traced.__wrapped__ = original  # type: ignore[attr-defined]
    return traced


def _iterate_sync(inner: Any, call_state: _CallState) -> Any:
    reply = _Reply()
    try:
        for chunk in inner:
            if call_state.first_chunk_mono is None:
                call_state.first_chunk_mono = time.monotonic()
            reply.feed(chunk)
            yield chunk
    except GeneratorExit:
        raise  # abandoned early: recorded as completed in finally
    except BaseException as exc:
        call_state.record_in_trace(reply, exc)
        raise
    finally:
        call_state.record_in_trace(reply)


async def _iterate_async(inner: Any, call_state: _CallState) -> Any:
    reply = _Reply()
    try:
        async for chunk in inner:
            if call_state.first_chunk_mono is None:
                call_state.first_chunk_mono = time.monotonic()
            reply.feed(chunk)
            yield chunk
    except GeneratorExit:
        raise
    except BaseException as exc:
        call_state.record_in_trace(reply, exc)
        raise
    finally:
        call_state.record_in_trace(reply)


def _traced_stream_sync(original: Any) -> Any:
    def traced(self: Any, *args: Any, **kwargs: Any) -> Any:
        if _framework_records_call():
            return original(self, *args, **kwargs)
        _record_tool_spans_for_request(kwargs)
        call_state = _CallState(self, kwargs, stream=True)
        try:
            inner = original(self, *args, **kwargs)
        except BaseException as exc:
            call_state.record_in_trace(None, exc)
            raise
        return _iterate_sync(inner, call_state)

    traced.__wrapped__ = original  # type: ignore[attr-defined]
    return traced


def _traced_stream_async(original: Any) -> Any:
    async def traced(self: Any, *args: Any, **kwargs: Any) -> Any:
        if _framework_records_call():
            return await original(self, *args, **kwargs)
        _record_tool_spans_for_request(kwargs)
        call_state = _CallState(self, kwargs, stream=True)
        try:
            inner = await original(self, *args, **kwargs)
        except BaseException as exc:
            call_state.record_in_trace(None, exc)
            raise
        return _iterate_async(inner, call_state)

    traced.__wrapped__ = original  # type: ignore[attr-defined]
    return traced


def _trace_scope_sync(original: Any) -> Any:
    """Public ``generate_content``: open one trace for the whole call.

    The per-request methods record the spans; without a trace opened here,
    each automatic-function-calling round trip would open its own trace and
    the tool span between them would be lost.
    """

    def generate_content(self: Any, *args: Any, **kwargs: Any) -> Any:
        if get_active_trace() is not None or _framework_records_call():
            return original(self, *args, **kwargs)
        set_active_framework(_FRAMEWORK_NAME)
        try:
            with _av_run(_SPAN_NAME):
                return original(self, *args, **kwargs)
        finally:
            clear_active_framework()

    generate_content.__wrapped__ = original  # type: ignore[attr-defined]
    return generate_content


def _trace_scope_async(original: Any) -> Any:
    async def generate_content(self: Any, *args: Any, **kwargs: Any) -> Any:
        if get_active_trace() is not None or _framework_records_call():
            return await original(self, *args, **kwargs)
        set_active_framework(_FRAMEWORK_NAME)
        try:
            with _av_run(_SPAN_NAME):
                return await original(self, *args, **kwargs)
        finally:
            clear_active_framework()

    generate_content.__wrapped__ = original  # type: ignore[attr-defined]
    return generate_content


def _wrapper_for(is_async: bool, is_stream: bool) -> Any:
    if is_stream:
        return _traced_stream_async if is_async else _traced_stream_sync
    return _traced_async if is_async else _traced_sync


# ---------------------------------------------------------------------------
# install() / uninstall()
# ---------------------------------------------------------------------------


def is_installed() -> bool:
    try:
        from google import genai  # type: ignore[import-not-found,attr-defined]
    except ImportError:
        return False
    return bool(getattr(genai, _INSTALL_FLAG, False))


@absorb_errors(fingerprint="morse_ai.adapters.google_genai.install")
def install() -> bool:
    """Patch ``google.genai`` so every client is traced. Idempotent.

    Returns True if installed (newly or already), False if ``google-genai``
    is not available or exposes none of the expected methods.
    """
    try:
        from google import genai  # type: ignore[import-not-found,attr-defined]
        from google.genai import (
            models as genai_models,  # type: ignore[import-not-found]
        )
    except ImportError:
        return False

    if getattr(genai, _INSTALL_FLAG, False):
        return True

    for cls_name, private, public, is_async, is_stream in _TARGETS:
        cls = getattr(genai_models, cls_name, None)
        if cls is None:
            continue
        attr = private if private in cls.__dict__ else public
        original = cls.__dict__.get(attr)
        if original is None:
            continue
        if attr == public:
            logger.debug(
                "amund: google-genai %s.%s missing; patching %s (AFC "
                "round trips are recorded as one span)",
                cls_name,
                private,
                public,
            )
        setattr(cls, attr, _wrapper_for(is_async, is_stream)(original))
        _patched.append((cls, attr, original))
        public_fn = cls.__dict__.get(public)
        if attr == private and not is_stream and public_fn is not None:
            # Streams are left alone: a trace opened around a generator would
            # close before its first chunk (each streamed hop records its own).
            scope = _trace_scope_async if is_async else _trace_scope_sync
            setattr(cls, public, scope(public_fn))
            _patched.append((cls, public, public_fn))

    if not _patched:
        logger.warning(
            "amund: google-genai adapter found no generate_content methods "
            "to patch; Gemini calls will run untraced"
        )
        return False

    setattr(genai, _INSTALL_FLAG, True)
    logger.info("amund: google-genai adapter installed")
    return True


@absorb_errors(fingerprint="morse_ai.adapters.google_genai.uninstall")
def uninstall() -> None:
    """Restore the original methods. Idempotent."""
    while _patched:
        cls, attr, original = _patched.pop()
        setattr(cls, attr, original)
    try:
        from google import genai  # type: ignore[import-not-found,attr-defined]
    except ImportError:
        return
    setattr(genai, _INSTALL_FLAG, False)
