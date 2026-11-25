"""Performance tests — verify <0.1ms blocking overhead for record()."""

import time

from amund.queue import TelemetryQueue


class TestPerformance:
    def test_queue_put_under_100_microseconds(self):
        """queue.put_nowait() must complete in <0.1ms (100 microseconds).

        This is the core performance requirement from the PRD (US-01).
        record() calls put_nowait() internally, so this is the blocking path.
        """
        q = TelemetryQueue(max_size=100_000)
        event = {
            "agent": "bench-agent",
            "success": True,
            "model": "claude-sonnet-4-6",
            "input_tokens": 1200,
            "output_tokens": 350,
            "cost_usd": 0.043,
        }

        iterations = 10_000
        start = time.perf_counter_ns()
        for _ in range(iterations):
            q.put(event)
        elapsed_ns = time.perf_counter_ns() - start

        mean_us = (elapsed_ns / iterations) / 1000
        assert mean_us < 100, f"put() mean={mean_us:.1f}us, must be <100us (0.1ms)"

    def test_record_dict_creation_under_100_microseconds(self):
        """RunRecord creation + to_dict() should be fast."""
        from amund.record import RunRecord

        iterations = 10_000
        start = time.perf_counter_ns()
        for _ in range(iterations):
            rec = RunRecord(
                agent="bench-agent",
                success=True,
                outcome="qualified",
                model="claude-sonnet-4-6",
                input_tokens=1200,
                output_tokens=350,
                cost_usd=0.043,
            )
            rec.to_dict()
        elapsed_ns = time.perf_counter_ns() - start

        mean_us = (elapsed_ns / iterations) / 1000
        # RunRecord + to_dict combined should be well under 100us
        assert mean_us < 100, f"record creation mean={mean_us:.1f}us, must be <100us"
