"""Tests for RunRecord dataclass."""

from importlib.metadata import version

from morse_ai.record import RunRecord


class TestRunRecord:
    def test_creates_with_defaults(self):
        rec = RunRecord(agent="test-agent")
        assert rec.agent == "test-agent"
        assert rec.status == "completed"
        # Asserted against the packaging metadata, not a literal: a hardcoded
        # version here would need hand-editing every release and would happily
        # pass while the shipped telemetry reported a stale version.
        assert rec.sdk_version == version("morse-ai")
        assert rec.run_id  # auto-generated
        assert rec.started_at  # auto-generated

    def test_to_dict_drops_nones(self):
        rec = RunRecord(agent="test-agent", success=True, cost_usd=0.043)
        d = rec.to_dict()
        assert "agent" in d
        assert "success" in d
        assert "cost_usd" in d
        assert "model" not in d  # None, should be dropped
        assert "error_message" not in d

    def test_to_dict_preserves_values(self):
        rec = RunRecord(
            agent="lead-qualifier",
            success=True,
            outcome="qualified",
            model="claude-sonnet-4-6",
            input_tokens=1200,
            output_tokens=350,
            cost_usd=0.043,
        )
        d = rec.to_dict()
        assert d["agent"] == "lead-qualifier"
        assert d["success"] is True
        assert d["outcome"] == "qualified"
        assert d["model"] == "claude-sonnet-4-6"
        assert d["input_tokens"] == 1200
        assert d["cost_usd"] == 0.043

    def test_unique_run_ids(self):
        r1 = RunRecord(agent="a")
        r2 = RunRecord(agent="a")
        assert r1.run_id != r2.run_id
