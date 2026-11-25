"""INFRACAP-07/08 — queue spans for Celery and SQS (boto3), inside the trace.

Not to be confused with ``amund.queue``, which is the SDK's own telemetry
buffer.

Producer side (default-on, active trace only)
    Publishing a Celery task or an SQS message inside an Amund trace records a
    ``queue`` span named ``publish <destination>`` carrying OTel messaging
    attributes in metadata (``messaging.system``, ``messaging.operation.type``,
    ``messaging.destination.name``), and injects a W3C ``traceparent`` into the
    message headers / message attributes. Message bodies, task arguments and
    keyword arguments are never read.

Consumer side (opt-in, ``AMUND_CAPTURE_QUEUE_CONSUMER=1``)
    A Celery worker continues the producer's trace from the ``traceparent`` and
    records a ``process <destination>`` span. For SQS, which has no worker
    framework to hook, use :func:`continue_from_sqs_message`. A missing or
    malformed ``traceparent`` starts a fresh trace and never raises.

Switches: ``AMUND_CAPTURE_QUEUE=0`` or ``AMUND_CAPTURE_INFRA=0`` turns the
producer side off (a specific flag wins).

The libraries are patched only if already imported, or right after they are
first imported — init never pays the import cost of a library the app does not use.
"""

from __future__ import annotations

import contextlib
import importlib.abc
import importlib.util
import logging
import sys
from typing import TYPE_CHECKING, Any

from amund._internal import env as _env
from amund.infra_capture import (
    _http_error_message as _class_name_only,
)
from amund.infra_capture import _infra_span, _SafeErrorSpan, capture_enabled
from amund.tracing import get_active_trace, run, span

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = logging.getLogger("amund")

TRACEPARENT = "traceparent"
_MAX_SQS_ATTRIBUTES = 10  # AWS limit per message

# patch key -> restore callable
_restores: dict[str, Any] = {}
_late_finder: _LateImportPatcher | None = None
# Producer spans on/off, decided once at init (not per call: reading os.environ
# on every publish is measurable). enable_queue_capture() turns it on.
_producer_on = True


def _consumer_enabled() -> bool:
    return _env.get("CAPTURE_QUEUE_CONSUMER").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _traceparent_now() -> str | None:
    from amund.propagation import get_current_traceparent

    return get_current_traceparent()


def _publish_span(system: str, destination: str | None, extra: dict | None = None):
    """A ``queue`` publish span with OTel messaging attributes."""
    name = f"publish {destination}" if destination else "publish"
    # Class name only on failure: broker errors can embed the connection URL,
    # credentials included.
    cm = _SafeErrorSpan(_infra_span("queue", name), _class_name_only)
    return _MessagingSpan(cm, system, "publish", destination, extra)


class _MessagingSpan:
    """Wrap a span context manager and stamp the messaging attributes on enter."""

    __slots__ = ("_cm", "_attrs")

    def __init__(
        self,
        cm: Any,
        system: str,
        operation: str,
        destination: str | None,
        extra: dict | None,
    ) -> None:
        self._cm = cm
        attrs: dict[str, Any] = {
            "messaging.system": system,
            "messaging.operation.type": operation,
        }
        if destination:
            attrs["messaging.destination.name"] = destination
        if extra:
            attrs.update(extra)
        self._attrs = attrs

    def __enter__(self) -> Any:
        sp = self._cm.__enter__()
        for k, v in self._attrs.items():
            sp.set_metadata(k, v)
        return sp

    def __exit__(self, *exc: Any) -> bool:
        return bool(self._cm.__exit__(*exc))


# ---------------------------------------------------------------------------
# Celery
# ---------------------------------------------------------------------------


def _celery_destination(app: Any, options: dict) -> str | None:
    queue = options.get("queue")
    if queue is not None:
        return str(getattr(queue, "name", queue))
    routing_key = options.get("routing_key")
    if routing_key:
        return str(routing_key)
    try:
        return str(app.conf.task_default_queue or "celery")
    except Exception:  # noqa: BLE001
        return "celery"


def patch_celery() -> bool:
    """Wrap ``Celery.send_task`` (the single path every ``apply_async`` takes to
    a broker). Idempotent."""
    if "celery" in _restores:
        return True
    try:
        from celery.app.base import Celery
    except Exception:  # noqa: BLE001
        return False

    original = Celery.send_task

    def send_task(self, name, args=None, kwargs=None, *a, **options):  # type: ignore[no-untyped-def]
        if get_active_trace() is None or not _producer_on:
            return original(self, name, args, kwargs, *a, **options)
        try:
            dest = _celery_destination(self, options)
            cm = _publish_span("celery", dest, {"celery.task.name": str(name)})
        except Exception:  # noqa: BLE001
            return original(self, name, args, kwargs, *a, **options)
        with cm:
            try:
                tp = _traceparent_now()
                if tp:
                    headers = dict(options.get("headers") or {})
                    headers.setdefault(TRACEPARENT, tp)
                    options["headers"] = headers
            except Exception as exc:  # noqa: BLE001
                logger.debug("amund: celery traceparent inject skipped: %s", exc)
            return original(self, name, args, kwargs, *a, **options)

    Celery.send_task = send_task
    _restores["celery"] = lambda: setattr(Celery, "send_task", original)
    _connect_celery_consumer()
    return True


_celery_runs: dict[str, Any] = {}


def _connect_celery_consumer() -> None:
    """Hook the worker signals. They exist always but do nothing unless
    ``AMUND_CAPTURE_QUEUE_CONSUMER`` is on at the moment a task starts."""
    try:
        from celery.signals import task_postrun, task_prerun
    except Exception:  # noqa: BLE001
        return

    def _prerun(sender=None, task_id=None, task=None, **kw):  # type: ignore[no-untyped-def]
        if not _consumer_enabled() or task_id is None:
            return
        try:
            request = getattr(task, "request", None)
            tp = getattr(request, TRACEPARENT, None)
            if tp is None:
                headers = getattr(request, "headers", None) or {}
                tp = headers.get(TRACEPARENT) if hasattr(headers, "get") else None
            dest = getattr(getattr(request, "delivery_info", None), "get", None)
            dest_name = dest("routing_key") if dest else None
            stack = contextlib.ExitStack()
            stack.enter_context(
                run(
                    f"celery:{getattr(task, 'name', 'task')}",
                    incoming_traceparent=tp if isinstance(tp, str) else None,
                )
            )
            stack.enter_context(
                _MessagingSpan(
                    span(f"process {dest_name}" if dest_name else "process"),
                    "celery",
                    "process",
                    dest_name,
                    {"celery.task.name": str(getattr(task, "name", ""))},
                )
            )
            _celery_runs[task_id] = stack
        except Exception as exc:  # noqa: BLE001
            logger.debug("amund: celery consumer continuation skipped: %s", exc)

    def _postrun(sender=None, task_id=None, **kw):  # type: ignore[no-untyped-def]
        stack = _celery_runs.pop(task_id, None)
        if stack is None:
            return
        try:
            stack.close()
        except Exception as exc:  # noqa: BLE001
            logger.debug("amund: celery consumer close skipped: %s", exc)

    task_prerun.connect(_prerun, weak=False, dispatch_uid="morse_ai_queue_prerun")
    task_postrun.connect(_postrun, weak=False, dispatch_uid="morse_ai_queue_postrun")

    def _disconnect() -> None:
        task_prerun.disconnect(dispatch_uid="morse_ai_queue_prerun")
        task_postrun.disconnect(dispatch_uid="morse_ai_queue_postrun")
        _celery_runs.clear()

    _restores["celery_signals"] = _disconnect


# ---------------------------------------------------------------------------
# SQS via botocore
# ---------------------------------------------------------------------------

_SQS_SEND = ("SendMessage", "SendMessageBatch")


def _queue_name_from_url(url: Any) -> str | None:
    try:
        name = str(url).rstrip("/").rsplit("/", 1)[-1]
        return name or None
    except Exception:  # noqa: BLE001
        return None


def _inject_sqs(operation: str, params: dict, tp: str) -> dict:
    """Copy of ``params`` with a traceparent message attribute on each message
    that has room for it. The original dict is never mutated."""
    attr = {"DataType": "String", "StringValue": tp}

    def with_attr(item: dict) -> dict:
        existing = dict(item.get("MessageAttributes") or {})
        if TRACEPARENT in existing or len(existing) >= _MAX_SQS_ATTRIBUTES:
            return item
        existing[TRACEPARENT] = attr
        return {**item, "MessageAttributes": existing}

    if operation == "SendMessage":
        return with_attr(params)
    entries = [with_attr(e) for e in params.get("Entries") or []]
    return {**params, "Entries": entries}


def patch_sqs() -> bool:
    """Wrap ``botocore.client.BaseClient._make_api_call`` for SQS send and
    (with the consumer flag) receive. Idempotent."""
    if "sqs" in _restores:
        return True
    try:
        from botocore.client import BaseClient
    except Exception:  # noqa: BLE001
        return False

    original = BaseClient._make_api_call

    def _make_api_call(self, operation_name, api_params):  # type: ignore[no-untyped-def]
        try:
            is_sqs = self.meta.service_model.service_name == "sqs"
        except Exception:  # noqa: BLE001
            is_sqs = False
        if not is_sqs:
            return original(self, operation_name, api_params)

        if operation_name == "ReceiveMessage" and _consumer_enabled():
            try:
                names = list(api_params.get("MessageAttributeNames") or [])
                if not any(n in ("All", ".*", TRACEPARENT) for n in names):
                    api_params = {
                        **api_params,
                        "MessageAttributeNames": [*names, TRACEPARENT],
                    }
            except Exception:  # noqa: BLE001
                pass
            return original(self, operation_name, api_params)

        if (
            operation_name not in _SQS_SEND
            or get_active_trace() is None
            or not _producer_on
        ):
            return original(self, operation_name, api_params)

        try:
            dest = _queue_name_from_url(api_params.get("QueueUrl"))
            cm = _publish_span("aws_sqs", dest)
        except Exception:  # noqa: BLE001
            return original(self, operation_name, api_params)
        with cm:
            try:
                tp = _traceparent_now()
                if tp:
                    api_params = _inject_sqs(operation_name, api_params, tp)
            except Exception as exc:  # noqa: BLE001
                logger.debug("amund: sqs traceparent inject skipped: %s", exc)
            return original(self, operation_name, api_params)

    BaseClient._make_api_call = _make_api_call
    _restores["sqs"] = lambda: setattr(BaseClient, "_make_api_call", original)
    return True


@contextlib.contextmanager
def continue_from_sqs_message(
    message: dict, agent_name: str = "sqs-consumer", queue_url: str | None = None
) -> Iterator[Any]:
    """Process one received SQS message inside the producer's trace.

    Requires ``AMUND_CAPTURE_QUEUE_CONSUMER=1``; otherwise it is a plain
    pass-through (``yield None``). A missing or malformed ``traceparent`` starts
    a fresh trace.

        for msg in sqs.receive_message(QueueUrl=url, ...)["Messages"]:
            with amund.queue_capture.continue_from_sqs_message(msg, queue_url=url):
                handle(msg)
    """
    if not _consumer_enabled():
        yield None
        return
    tp = None
    try:
        attrs = message.get("MessageAttributes") or {}
        tp = (attrs.get(TRACEPARENT) or {}).get("StringValue")
        if not isinstance(tp, str):
            tp = None
    except Exception:  # noqa: BLE001
        tp = None
    dest = _queue_name_from_url(queue_url) if queue_url else None
    with (
        run(agent_name, incoming_traceparent=tp) as r,
        _MessagingSpan(
            span(f"process {dest}" if dest else "process"),
            "aws_sqs",
            "process",
            dest,
            None,
        ),
    ):
        yield r


# ---------------------------------------------------------------------------
# enable / disable
# ---------------------------------------------------------------------------


class _LateImportPatcher(importlib.abc.MetaPathFinder):
    """Runs a patch function right after a not-yet-imported module is imported."""

    def __init__(self, targets: dict[str, Any]) -> None:
        self._targets = dict(targets)

    def find_spec(self, fullname, path, target=None):  # type: ignore[no-untyped-def]
        patcher = self._targets.get(fullname)
        if patcher is None:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(fullname)
        except Exception:  # noqa: BLE001
            spec = None
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return None
        loader = spec.loader
        original_exec = getattr(loader, "exec_module", None)
        if original_exec is None:
            return None

        def exec_module(module):  # type: ignore[no-untyped-def]
            original_exec(module)
            self._targets.pop(fullname, None)
            try:
                patcher()
            except Exception as exc:  # noqa: BLE001
                logger.debug("amund: late queue patch %s skipped: %s", fullname, exc)

        loader.exec_module = exec_module  # type: ignore[method-assign]
        return spec


def enable_queue_capture() -> list[str]:
    """Patch Celery / SQS now if imported, otherwise right after first import.
    Idempotent. Returns the integrations patched immediately."""
    global _late_finder
    patched: list[str] = []
    late: dict[str, Any] = {}
    if "celery.app.base" in sys.modules:
        if patch_celery():
            patched.append("celery")
    else:
        late["celery.app.base"] = patch_celery
    if "botocore.client" in sys.modules:
        if patch_sqs():
            patched.append("sqs")
    else:
        late["botocore.client"] = patch_sqs
    if late and _late_finder is None:
        _late_finder = _LateImportPatcher(late)
        sys.meta_path.insert(0, _late_finder)
    return patched


def disable_queue_capture() -> None:
    """Undo every patch and the late-import hook. For tests."""
    global _late_finder, _producer_on
    _producer_on = True
    for restore in list(_restores.values()):
        with contextlib.suppress(Exception):
            restore()
    _restores.clear()
    if _late_finder is not None:
        with contextlib.suppress(ValueError):
            sys.meta_path.remove(_late_finder)
        _late_finder = None


def maybe_enable_from_env() -> None:
    """Called from SDK init. On unless ``AMUND_CAPTURE_QUEUE=0`` /
    ``AMUND_CAPTURE_INFRA=0``. The consumer flag is read when a task starts."""
    global _producer_on
    _producer_on = capture_enabled("AMUND_CAPTURE_QUEUE")
    if _producer_on or _consumer_enabled():
        enable_queue_capture()
