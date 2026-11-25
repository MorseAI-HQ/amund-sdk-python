"""INFRACAP-09 — canary PII audit for default-on infra capture.

Canary values are pushed through every instrumented path: SQL bound parameters,
SQL literals written inline, URL path and query, request headers, request and
response bodies, queue message bodies and arguments, and the error text of calls
that fail. The trace is then exported through the real OTLP attribute mapping
and every canary is searched for in names, attributes, events and status.

Redaction is switched OFF for the audit on purpose. With it on, the generic
email/card patterns would mask some canaries and the test could pass because of
redaction rather than because capture never read the value.
"""

import json
import sqlite3

import httpx
import pytest
import requests

from amund import infra_capture as ic
from amund import queue_capture as qc
from amund._internal.redaction import RedactionConfig
from amund.otel_provider import emit_trace_entry, get_engine
from amund.tracing import _get_last_trace, run

celery = pytest.importorskip("celery")
boto3 = pytest.importorskip("boto3")

EMAIL = "canary.user@pii-audit.test"
CARD = "4111111111111111"
TOKEN = "tok_canary_9f3a7c"
BEARER = "sk-canary-bearer-0123456789"
COOKIE = "session=canary-cookie-777"
BODY = "canary-body-payload-42"
ARG = "canary-task-argument"
CANARIES = [EMAIL, CARD, TOKEN, BEARER, COOKIE, BODY, ARG, "canarypass"]


@pytest.fixture
def engine(monkeypatch):
    for name in (
        "MORSE_CAPTURE_INFRA",
        "MORSE_CAPTURE_DB",
        "MORSE_CAPTURE_HTTP",
        "MORSE_CAPTURE_QUEUE",
    ):
        monkeypatch.delenv(name, raising=False)
    eng = get_engine()
    eng.configure(
        endpoint="http://127.0.0.1:1/api/v1/otlp/v1/traces",
        api_key="av_test",  # pragma: allowlist secret
        sample_rate=1.0,
        schedule_delay_millis=50,
        max_queue_size=4096,
        max_export_batch_size=512,
        max_field_bytes=65536,
        timeout=1.0,
        redaction_config=RedactionConfig.build(enabled=False),
        capture_for_tests=True,
    )
    ic.disable_db_capture()
    ic.disable_http_capture()
    qc.disable_queue_capture()
    ic.maybe_enable_from_env()
    qc.maybe_enable_from_env()
    yield eng
    ic.disable_db_capture()
    ic.disable_http_capture()
    qc.disable_queue_capture()
    eng.reset_for_testing()


def _everything_exported(engine) -> str:
    """Names, attributes, events and status of every exported span, as one string."""
    engine.flush()
    out = []
    for s in engine.recorder.spans:
        out.append(
            {
                "name": s.name,
                "attributes": dict(s.attributes or {}),
                "events": [
                    {"name": e.name, "attributes": dict(e.attributes or {})}
                    for e in s.events
                ],
                "status": str(s.status.description),
            }
        )
    return json.dumps(out, default=str)


class _FailingAdapter(requests.adapters.BaseAdapter):
    def send(self, request, **kwargs):  # type: ignore[no-untyped-def]
        raise requests.ConnectionError(
            f"connection failed for url: {request.url}"  # message carries the URL
        )

    def close(self):  # type: ignore[no-untyped-def]
        return None


class _OkAdapter(requests.adapters.BaseAdapter):
    def send(self, request, **kwargs):  # type: ignore[no-untyped-def]
        resp = requests.Response()
        resp.status_code = 200
        resp._content = BODY.encode()
        resp.request = request
        return resp

    def close(self):  # type: ignore[no-untyped-def]
        return None


@pytest.mark.req("INFRACAP-09")
def test_zero_canaries_in_exported_spans(engine):
    from botocore.stub import Stubber

    # --- Celery: unreachable broker so the publish also fails -------------
    app = celery.Celery("pii", broker="amqp://user:canarypass@127.0.0.1:1//")
    app.conf.broker_connection_retry_on_startup = False

    @app.task(name="pii.work")
    def work(*a, **kw):  # pragma: no cover
        return None

    sqs = boto3.client(
        "sqs",
        region_name="us-east-1",
        aws_access_key_id="test",
        aws_secret_access_key="test",  # pragma: allowlist secret
    )
    stub = Stubber(sqs)
    stub.add_response("send_message", {"MessageId": "m1", "MD5OfMessageBody": "x"})
    stub.add_client_error("send_message", service_message=f"nope {BODY}")

    with stub, run("pii-agent"):
        # SQL: bound params, inline literals, and a failing statement
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE users (email TEXT UNIQUE, card TEXT, n INTEGER)")
        conn.execute("INSERT INTO users VALUES (?, ?, ?)", (EMAIL, CARD, 1))
        conn.execute(f"SELECT * FROM users WHERE email = '{EMAIL}' AND n = {CARD}")
        conn.executemany("INSERT INTO users VALUES (?, ?, ?)", [(TOKEN, BODY, 2)])
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(f"INSERT INTO users VALUES ('{EMAIL}', '{CARD}', 3)")
        conn.execute(f"SELECT 1 -- {BEARER}\n")

        # HTTP via httpx: query, headers, body, failing transport
        def handler(request: httpx.Request) -> httpx.Response:
            if "/fail" in request.url.path:
                raise httpx.ConnectError(f"cannot reach {request.url}")
            return httpx.Response(500, text=BODY)

        client = httpx.Client(transport=httpx.MockTransport(handler))
        url = f"https://api.pii-audit.test/users/orders?email={EMAIL}&token={TOKEN}&card={CARD}"
        client.post(
            url,
            headers={"Authorization": f"Bearer {BEARER}", "Cookie": COOKIE},
            content=BODY,
        )
        with pytest.raises(httpx.ConnectError):
            client.get(f"https://api.pii-audit.test/fail?email={EMAIL}&token={TOKEN}")

        # HTTP via requests
        session = requests.Session()
        session.mount("https://ok.pii-audit.test", _OkAdapter())
        session.mount("https://bad.pii-audit.test", _FailingAdapter())
        session.get(
            f"https://ok.pii-audit.test/path?email={EMAIL}&token={TOKEN}",
            headers={"Authorization": f"Bearer {BEARER}"},
        )
        with pytest.raises(requests.ConnectionError):
            session.get(f"https://bad.pii-audit.test/path?email={EMAIL}&token={TOKEN}")

        # Celery publish (task args and a custom header carry canaries)
        with pytest.raises(Exception):  # noqa: B017, PT011 - broker is unreachable
            work.apply_async(
                args=(ARG, EMAIL),
                kwargs={"card": CARD},
                headers={"x-note": COOKIE},
                retry=False,
            )

        # SQS publish: success and a failing call
        sqs.send_message(
            QueueUrl="https://sqs.us-east-1.amazonaws.com/123456789012/pii-queue",
            MessageBody=BODY,
            MessageAttributes={"who": {"DataType": "String", "StringValue": EMAIL}},
        )
        with pytest.raises(Exception):  # noqa: B017, PT011 - stubbed client error
            sqs.send_message(
                QueueUrl="https://sqs.us-east-1.amazonaws.com/123456789012/pii-queue",
                MessageBody=BODY,
            )

    trace = _get_last_trace()
    emit_trace_entry(trace)

    # The audit is only meaningful if every path really produced a span.
    kinds = [s["type"] for s in trace["spans"]]
    assert kinds.count("db") >= 6
    assert kinds.count("http") >= 4
    queue_spans = [
        s for s in trace["spans"] if (s.get("metadata") or {}).get("messaging.system")
    ]
    assert len(queue_spans) >= 3

    haystacks = {
        "trace payload": json.dumps(trace, default=str),
        "exported OTLP spans": _everything_exported(engine),
    }
    hits = {
        (where, canary)
        for where, text in haystacks.items()
        for canary in CANARIES
        if canary in text
    }
    assert hits == set(), f"canary values leaked: {sorted(hits)}"
