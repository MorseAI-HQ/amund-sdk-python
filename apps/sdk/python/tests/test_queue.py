"""Tests for TelemetryQueue."""

from morse_ai.queue import SHUTDOWN, TelemetryQueue


class TestTelemetryQueue:
    def test_put_and_drain(self):
        q = TelemetryQueue(max_size=100)
        q.put({"type": "test1"})
        q.put({"type": "test2"})
        batch = q.drain(max_batch=10)
        assert len(batch) == 2
        assert batch[0]["type"] == "test1"

    def test_drain_max_batch(self):
        q = TelemetryQueue(max_size=100)
        for i in range(50):
            q.put({"n": i})
        batch = q.drain(max_batch=10)
        assert len(batch) == 10

    def test_drain_empty_queue(self):
        q = TelemetryQueue()
        batch = q.drain()
        assert batch == []

    def test_put_full_queue_drops(self):
        q = TelemetryQueue(max_size=2)
        assert q.put({"a": 1}) is True
        assert q.put({"b": 2}) is True
        assert q.put({"c": 3}) is False  # dropped

    def test_shutdown_sentinel(self):
        q = TelemetryQueue()
        q.put({"data": 1})
        q.signal_shutdown()
        batch = q.drain(max_batch=100)
        assert len(batch) == 2
        assert SHUTDOWN in batch

    def test_size_and_empty(self):
        q = TelemetryQueue()
        assert q.empty is True
        assert q.size == 0
        q.put({"x": 1})
        assert q.empty is False
        assert q.size == 1
