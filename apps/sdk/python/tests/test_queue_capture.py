"""INFRACAP-07/08 — queue spans for Celery and SQS."""

import pytest

from amund import queue_capture as qc
from amund.propagation import get_current_traceparent
from amund.tracing import _get_last_trace, run

celery = pytest.importorskip("celery")
boto3 = pytest.importorskip("boto3")

_ENV = (
    "MORSE_CAPTURE_INFRA",
    "MORSE_CAPTURE_QUEUE",
    "MORSE_CAPTURE_QUEUE_CONSUMER",
)
BAD = "garbage-not-a-traceparent"


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for name in _ENV:
        monkeypatch.delenv(name, raising=False)
    qc.disable_queue_capture()
    yield
    qc.disable_queue_capture()


def _queue_spans():
    return [
        s
        for s in _get_last_trace()["spans"]
        if (s.get("metadata") or {}).get("messaging.system")
    ]


# --- Celery ----------------------------------------------------------------


@pytest.fixture
def celery_app():
    app = celery.Celery("t", broker="memory://", backend="cache+memory://")
    app.conf.task_default_queue = "agents"

    @app.task(name="t.work")
    def work(x):  # pragma: no cover - never executed, only published
        return x

    app._work = work
    return app


def _published_headers(app, queue="agents"):
    with app.connection_for_read() as conn:
        simple = conn.SimpleQueue(queue)
        try:
            return simple.get(block=False).headers
        finally:
            simple.close()


@pytest.mark.req("INFRACAP-07")
def test_celery_publish_inside_trace_records_span_and_injects_traceparent(celery_app):
    qc.maybe_enable_from_env()
    with run("agent") as r:
        tid = r.trace_id
        celery_app._work.apply_async(args=("secret-arg-canary",))
        expected_prefix = f"00-{tid}-"
    spans = _queue_spans()
    assert [s["name"] for s in spans] == ["publish agents"]
    meta = spans[0]["metadata"]
    assert meta["messaging.system"] == "celery"
    assert meta["messaging.operation.type"] == "publish"
    assert meta["messaging.destination.name"] == "agents"
    assert spans[0]["type"] == "queue"
    assert "secret-arg-canary" not in str(_get_last_trace())
    headers = _published_headers(celery_app)
    assert headers["traceparent"].startswith(expected_prefix)


@pytest.mark.req("INFRACAP-07")
def test_celery_publish_outside_trace_records_nothing_and_does_not_inject(celery_app):
    qc.maybe_enable_from_env()
    celery_app._work.apply_async(args=(1,))
    assert "traceparent" not in _published_headers(celery_app)


@pytest.mark.req("INFRACAP-07")
@pytest.mark.parametrize("env", ["MORSE_CAPTURE_QUEUE", "MORSE_CAPTURE_INFRA"])
def test_celery_kill_switches(monkeypatch, celery_app, env):
    monkeypatch.setenv(env, "0")
    qc.maybe_enable_from_env()
    with run("agent"):
        celery_app._work.apply_async(args=(1,))
    assert _queue_spans() == []
    assert "traceparent" not in _published_headers(celery_app)


@pytest.mark.req("INFRACAP-07")
def test_celery_explicit_queue_option_names_the_destination(celery_app):
    qc.maybe_enable_from_env()
    with run("agent"):
        celery_app._work.apply_async(args=(1,), queue="other")
    assert _queue_spans()[0]["name"] == "publish other"


def _fake_task(celery_app, traceparent):
    task = celery_app._work
    task.push_request(
        id="task-1",
        traceparent=traceparent,
        delivery_info={"routing_key": "agents"},
    )
    return task


@pytest.mark.req("INFRACAP-08")
def test_celery_consumer_is_off_without_flag(celery_app):
    from celery.signals import task_postrun, task_prerun

    qc.maybe_enable_from_env()
    task = _fake_task(celery_app, "00-" + "a" * 32 + "-" + "b" * 16 + "-01")
    task_prerun.send(sender=task, task_id="task-1", task=task)
    task_postrun.send(sender=task, task_id="task-1", task=task)
    assert qc._celery_runs == {}


@pytest.mark.req("INFRACAP-08")
def test_celery_consumer_continues_producer_trace(monkeypatch, celery_app):
    from celery.signals import task_postrun, task_prerun

    monkeypatch.setenv("MORSE_CAPTURE_QUEUE_CONSUMER", "1")
    qc.maybe_enable_from_env()
    producer_trace = "c" * 32
    task = _fake_task(celery_app, f"00-{producer_trace}-{'d' * 16}-01")
    task_prerun.send(sender=task, task_id="task-1", task=task)
    task_postrun.send(sender=task, task_id="task-1", task=task)
    last = _get_last_trace()
    assert last["trace"]["trace_id"] == producer_trace
    proc = [s for s in last["spans"] if s["name"] == "process agents"]
    assert proc
    assert proc[0]["metadata"]["messaging.operation.type"] == "process"


@pytest.mark.req("INFRACAP-08")
@pytest.mark.parametrize("tp", [BAD, None, ""])
def test_celery_consumer_bad_or_missing_traceparent_starts_fresh_trace(
    monkeypatch, celery_app, tp
):
    from celery.signals import task_postrun, task_prerun

    monkeypatch.setenv("MORSE_CAPTURE_QUEUE_CONSUMER", "1")
    qc.maybe_enable_from_env()
    task = _fake_task(celery_app, tp)
    task_prerun.send(sender=task, task_id="task-1", task=task)
    task_postrun.send(sender=task, task_id="task-1", task=task)
    assert len(_get_last_trace()["trace"]["trace_id"]) == 32


# --- SQS -------------------------------------------------------------------

QUEUE_URL = "https://sqs.us-east-1.amazonaws.com/123456789012/agent-jobs"


@pytest.fixture
def sqs():
    from botocore.stub import Stubber

    client = boto3.client(
        "sqs",
        region_name="us-east-1",
        aws_access_key_id="test",
        aws_secret_access_key="test",  # pragma: allowlist secret
    )
    sent: list[dict] = []
    client.meta.events.register(
        "before-parameter-build.sqs.*", lambda params, **kw: sent.append(dict(params))
    )
    stub = Stubber(client)
    with stub:
        yield client, stub, sent


def _send_ok(stub, *, batch=False):
    if batch:
        stub.add_response("send_message_batch", {"Successful": [], "Failed": []})
    else:
        stub.add_response("send_message", {"MessageId": "m1", "MD5OfMessageBody": "x"})


@pytest.mark.req("INFRACAP-07")
def test_sqs_send_records_span_and_injects_attribute(sqs):
    client, stub, sent = sqs
    qc.maybe_enable_from_env()
    _send_ok(stub)
    with run("agent") as r:
        client.send_message(QueueUrl=QUEUE_URL, MessageBody="body-canary")
        prefix = f"00-{r.trace_id}-"
    spans = _queue_spans()
    assert spans[0]["name"] == "publish agent-jobs"
    assert spans[0]["metadata"]["messaging.system"] == "aws_sqs"
    assert "body-canary" not in str(_get_last_trace())
    tp = sent[0]["MessageAttributes"]["traceparent"]
    assert tp["DataType"] == "String"
    assert tp["StringValue"].startswith(prefix)


@pytest.mark.req("INFRACAP-07")
def test_sqs_batch_injects_each_entry(sqs):
    client, stub, sent = sqs
    qc.maybe_enable_from_env()
    _send_ok(stub, batch=True)
    with run("agent"):
        client.send_message_batch(
            QueueUrl=QUEUE_URL,
            Entries=[{"Id": "1", "MessageBody": "a"}, {"Id": "2", "MessageBody": "b"}],
        )
    entries = sent[0]["Entries"]
    assert all("traceparent" in e["MessageAttributes"] for e in entries)


@pytest.mark.req("INFRACAP-07")
def test_sqs_full_attribute_set_is_left_alone(sqs):
    client, stub, sent = sqs
    qc.maybe_enable_from_env()
    _send_ok(stub)
    attrs = {f"a{i}": {"DataType": "String", "StringValue": "v"} for i in range(10)}
    with run("agent"):
        client.send_message(
            QueueUrl=QUEUE_URL, MessageBody="b", MessageAttributes=attrs
        )
    assert "traceparent" not in sent[0]["MessageAttributes"]


@pytest.mark.req("INFRACAP-07")
def test_sqs_outside_trace_is_untouched(sqs):
    client, stub, sent = sqs
    qc.maybe_enable_from_env()
    _send_ok(stub)
    client.send_message(QueueUrl=QUEUE_URL, MessageBody="b")
    assert "MessageAttributes" not in sent[0]


@pytest.mark.req("INFRACAP-07")
def test_sqs_kill_switch(monkeypatch, sqs):
    client, stub, sent = sqs
    monkeypatch.setenv("MORSE_CAPTURE_INFRA", "0")
    qc.maybe_enable_from_env()
    _send_ok(stub)
    with run("agent"):
        client.send_message(QueueUrl=QUEUE_URL, MessageBody="b")
    assert _queue_spans() == []


@pytest.mark.req("INFRACAP-08")
def test_sqs_receive_requests_traceparent_only_with_consumer_flag(monkeypatch, sqs):
    client, stub, sent = sqs
    qc.maybe_enable_from_env()
    stub.add_response("receive_message", {})
    client.receive_message(QueueUrl=QUEUE_URL)
    assert "MessageAttributeNames" not in sent[0]

    monkeypatch.setenv("MORSE_CAPTURE_QUEUE_CONSUMER", "1")
    stub.add_response("receive_message", {})
    client.receive_message(QueueUrl=QUEUE_URL)
    assert sent[1]["MessageAttributeNames"] == ["traceparent"]


@pytest.mark.req("INFRACAP-08")
def test_continue_from_sqs_message_joins_the_producer_trace(monkeypatch):
    monkeypatch.setenv("MORSE_CAPTURE_QUEUE_CONSUMER", "1")
    trace = "e" * 32
    msg = {
        "MessageAttributes": {
            "traceparent": {"StringValue": f"00-{trace}-{'f' * 16}-01"}
        }
    }
    with qc.continue_from_sqs_message(msg, queue_url=QUEUE_URL):
        assert get_current_traceparent().startswith(f"00-{trace}-")
    last = _get_last_trace()
    assert last["trace"]["trace_id"] == trace
    assert any(s["name"] == "process agent-jobs" for s in last["spans"])


@pytest.mark.req("INFRACAP-08")
@pytest.mark.parametrize(
    "msg",
    [
        {},
        {"MessageAttributes": {"traceparent": {"StringValue": BAD}}},
        {"MessageAttributes": "not-a-dict"},
    ],
)
def test_continue_from_sqs_message_tolerates_bad_input(monkeypatch, msg):
    monkeypatch.setenv("MORSE_CAPTURE_QUEUE_CONSUMER", "1")
    with qc.continue_from_sqs_message(msg, queue_url=QUEUE_URL):
        pass
    assert len(_get_last_trace()["trace"]["trace_id"]) == 32


@pytest.mark.req("INFRACAP-08")
def test_continue_from_sqs_message_is_passthrough_without_flag():
    with qc.continue_from_sqs_message({}) as r:
        assert r is None


@pytest.mark.req("INFRACAP-07")
def test_late_import_patcher_runs_after_first_import(tmp_path, monkeypatch):
    import sys

    (tmp_path / "morse_late_probe.py").write_text("VALUE = 1\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop("morse_late_probe", None)
    calls: list[bool] = []
    finder = qc._LateImportPatcher(
        {"morse_late_probe": lambda: calls.append("morse_late_probe" in sys.modules)}
    )
    sys.meta_path.insert(0, finder)
    try:
        assert calls == []
        import morse_late_probe  # noqa: F401
    finally:
        sys.meta_path.remove(finder)
        sys.modules.pop("morse_late_probe", None)
    assert calls == [True]
