"""Tests for the completion monitor guard in pipe_executor.

The guard must use the same attribute name as Leg.__init__ to prevent
double-starting the monitor thread."""
import inspect

from speech_pipeline.telephony.leg import Leg
from speech_pipeline.telephony.pipe_executor import CallPipeExecutor


class TestCompletionMonitorGuard:

    def test_leg_initializes_flag_without_underscore(self):
        leg = Leg("l1", "inbound", "+49123", "pbx1", "sub1")
        assert hasattr(leg, "completion_monitor_started")
        assert leg.completion_monitor_started is False

    def test_guard_reads_correct_flag(self):
        """When the flag is True, getattr must see it."""
        leg = Leg("l1", "inbound", "+49123", "pbx1", "sub1")
        leg.completion_monitor_started = True
        assert getattr(leg, "completion_monitor_started", False) is True

    def test_pipe_executor_uses_matching_attribute_name(self):
        """Source-level check: _start_sip_monitors must NOT use the
        underscore-prefixed variant that would silently always be False."""
        source = inspect.getsource(CallPipeExecutor._start_sip_monitors)
        assert "_completion_monitor_started" not in source, \
            "pipe_executor uses wrong attribute name _completion_monitor_started"
        assert "completion_monitor_started" in source
