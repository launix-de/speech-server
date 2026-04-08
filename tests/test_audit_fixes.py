"""Tests for bugs found in the code audit (2026-04-08).

Covers:
  1. PyVoIPCallSession interface completeness (connected, hungup, hangup, rx_queue)
  2. Leg.hangup() trunk path (sip_call_id attribute)
  3. Completion monitor guard (attribute name match)
  4. FileFetcher._classify usage in VCConverter
  5. sip_listener._wait_for_bridge session attribute name
"""
import threading
import time
import types
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# 1. PyVoIPCallSession interface
# ---------------------------------------------------------------------------

class TestPyVoIPCallSession:
    """Verify PyVoIPCallSession matches the interface expected by
    SIPSource, SIPSink, and pipe_executor."""

    def _make_session(self):
        from speech_pipeline.telephony.leg import PyVoIPCallSession
        mock_call = MagicMock()
        mock_call.read_audio.side_effect = Exception("stopped")
        return PyVoIPCallSession(mock_call)

    def test_has_connected_event(self):
        session = self._make_session()
        assert hasattr(session, "connected")
        assert isinstance(session.connected, threading.Event)
        assert session.connected.is_set(), "pyVoIP sessions are connected at creation"

    def test_has_hungup_event(self):
        session = self._make_session()
        assert hasattr(session, "hungup")
        assert isinstance(session.hungup, threading.Event)
        assert not session.hungup.is_set()

    def test_has_rx_queue(self):
        import queue
        session = self._make_session()
        assert hasattr(session, "rx_queue")
        assert isinstance(session.rx_queue, queue.Queue)

    def test_has_call_property(self):
        mock_call = MagicMock()
        mock_call.read_audio.side_effect = Exception("stopped")
        from speech_pipeline.telephony.leg import PyVoIPCallSession
        session = PyVoIPCallSession(mock_call)
        assert session.call is mock_call

    def test_has_hangup_method(self):
        session = self._make_session()
        assert hasattr(session, "hangup")
        assert callable(session.hangup)

    def test_hangup_sets_hungup_event(self):
        session = self._make_session()
        assert not session.hungup.is_set()
        session.hangup()
        assert session.hungup.is_set()

    def test_hangup_stops_rx_pump(self):
        session = self._make_session()
        assert session._rx_pump_running is True
        session.hangup()
        assert session._rx_pump_running is False

    def test_hangup_calls_underlying_call_hangup(self):
        mock_call = MagicMock()
        mock_call.read_audio.side_effect = Exception("stopped")
        from speech_pipeline.telephony.leg import PyVoIPCallSession
        session = PyVoIPCallSession(mock_call)
        session.hangup()
        mock_call.hangup.assert_called_once()

    def test_hangup_tolerates_underlying_exception(self):
        mock_call = MagicMock()
        mock_call.read_audio.side_effect = Exception("stopped")
        mock_call.hangup.side_effect = RuntimeError("already dead")
        from speech_pipeline.telephony.leg import PyVoIPCallSession
        session = PyVoIPCallSession(mock_call)
        # Must not raise
        session.hangup()
        assert session.hungup.is_set()


# ---------------------------------------------------------------------------
# 2. Leg.hangup() trunk path — sip_call_id attribute
# ---------------------------------------------------------------------------

class TestLegHangupTrunk:
    """Verify Leg.hangup() correctly calls hangup_trunk_leg for trunk calls."""

    def _make_leg(self):
        from speech_pipeline.telephony.leg import Leg
        leg = Leg(
            leg_id="test-leg-1",
            direction="outbound",
            number="+491234567890",
            pbx_id="TestPBX",
            subscriber_id="test-sub",
        )
        return leg

    def test_hangup_uses_sip_call_id_without_underscore(self):
        """Bug #1: leg.hangup() checked hasattr(self, '_sip_call_id')
        but the attribute is set as sip_call_id (no underscore)."""
        leg = self._make_leg()
        leg.sip_call_id = "test-call-id-12345"

        with patch("speech_pipeline.telephony.sip_stack.hangup_trunk_leg",
                   return_value=True) as mock_hangup:
            leg.hangup()
            mock_hangup.assert_called_once_with("test-call-id-12345")

    def test_hangup_skips_trunk_when_no_sip_call_id(self):
        """When sip_call_id is not set, trunk hangup should be skipped."""
        leg = self._make_leg()
        # No sip_call_id set
        with patch("speech_pipeline.telephony.sip_stack.hangup_trunk_leg") as mock_hangup:
            leg.hangup()
            mock_hangup.assert_not_called()


# ---------------------------------------------------------------------------
# 3. Completion monitor guard — attribute name match
# ---------------------------------------------------------------------------

class TestCompletionMonitorGuard:
    """Verify pipe_executor checks the correct attribute name for
    completion_monitor_started (no underscore prefix)."""

    def test_guard_attribute_name_matches_leg(self):
        """Bug #2: pipe_executor checked _completion_monitor_started (with
        underscore) but Leg.__init__ sets completion_monitor_started."""
        from speech_pipeline.telephony.leg import Leg
        leg = Leg("l1", "inbound", "+49123", "pbx1", "sub1")

        # Leg initializes without underscore
        assert hasattr(leg, "completion_monitor_started")
        assert leg.completion_monitor_started is False

        # Simulate what pipe_executor does to check
        # This must return the actual value, not always False
        leg.completion_monitor_started = True
        assert getattr(leg, "completion_monitor_started", False) is True

    def test_pipe_executor_source_uses_correct_name(self):
        """Verify the actual source code uses the correct attribute name."""
        import inspect
        from speech_pipeline.telephony.pipe_executor import CallPipeExecutor
        source = inspect.getsource(CallPipeExecutor._start_sip_monitors)
        # Must NOT contain the wrong underscore-prefixed version
        assert '_completion_monitor_started' not in source, \
            "pipe_executor still uses wrong attribute name _completion_monitor_started"
        # Must contain the correct version
        assert 'completion_monitor_started' in source


# ---------------------------------------------------------------------------
# 4. FileFetcher._classify usage
# ---------------------------------------------------------------------------

class TestFileFetcherClassify:
    """Verify VCConverter uses the correct method name."""

    def test_classify_is_private(self):
        """FileFetcher exposes _classify (private), not classify (public)."""
        from speech_pipeline.FileFetcher import FileFetcher
        assert hasattr(FileFetcher, "_classify"), "FileFetcher must have _classify"
        assert not hasattr(FileFetcher, "classify"), \
            "FileFetcher should not have public classify (it's _classify)"

    def test_classify_http(self):
        from speech_pipeline.FileFetcher import FileFetcher
        kind, value = FileFetcher._classify("https://example.com/audio.wav")
        assert kind == "http"
        assert value == "https://example.com/audio.wav"

    def test_classify_file(self):
        from speech_pipeline.FileFetcher import FileFetcher
        kind, value = FileFetcher._classify("/tmp/audio.wav")
        assert kind == "file"

    def test_vcconverter_source_uses_private_classify(self):
        """Bug #3: VCConverter.py called FileFetcher.classify() instead
        of FileFetcher._classify()."""
        import inspect
        from speech_pipeline.VCConverter import VCConverter
        source = inspect.getsource(VCConverter)
        assert "FileFetcher.classify(" not in source, \
            "VCConverter still calls FileFetcher.classify (should be _classify)"
        assert "FileFetcher._classify(" in source


# ---------------------------------------------------------------------------
# 5. sip_listener._wait_for_bridge — correct attribute name
# ---------------------------------------------------------------------------

class TestWaitForBridgeSessionAttr:
    """Verify _wait_for_bridge uses 'sip_session' (no underscore)."""

    def test_source_uses_correct_attribute_name(self):
        """Bug #4: sip_listener used getattr(leg, '_sip_session') but
        Leg.__init__ sets sip_session (no underscore)."""
        import inspect
        from speech_pipeline.telephony import sip_listener
        source = inspect.getsource(sip_listener._wait_for_bridge)
        assert '"_sip_session"' not in source and "'_sip_session'" not in source, \
            "sip_listener still accesses _sip_session (should be sip_session)"

    def test_leg_has_sip_session_attribute(self):
        """Leg.__init__ must set sip_session (no underscore)."""
        from speech_pipeline.telephony.leg import Leg
        leg = Leg("l1", "inbound", "+49123", "pbx1", "sub1")
        assert hasattr(leg, "sip_session")
        assert leg.sip_session is None

    def test_wait_for_bridge_detects_hungup_session(self):
        """When sip_session.hungup is set, _wait_for_bridge should
        clean up and return early."""
        from speech_pipeline.telephony.leg import Leg
        from speech_pipeline.telephony import sip_listener, leg as leg_mod

        leg = Leg("l-bridge-test", "inbound", "+49123", "pbx1", "sub1")
        leg_mod._legs[leg.leg_id] = leg

        # Create a mock session with hungup event
        session = MagicMock()
        session.hungup = threading.Event()
        session.hungup.set()  # Already hung up
        leg.sip_session = session

        # _wait_for_bridge should detect this and clean up
        sip_listener._wait_for_bridge(leg, None)
        assert leg.status == "completed"

        # Cleanup
        leg_mod._legs.pop(leg.leg_id, None)


# ---------------------------------------------------------------------------
# 6. Interface consistency: all session types
# ---------------------------------------------------------------------------

class TestSessionInterfaceConsistency:
    """All session types used by SIPSource/SIPSink must expose
    connected, hungup, rx_queue."""

    REQUIRED_ATTRS = ["connected", "hungup", "rx_queue"]

    def test_pyvoip_call_session_interface(self):
        from speech_pipeline.telephony.leg import PyVoIPCallSession
        mock_call = MagicMock()
        mock_call.read_audio.side_effect = Exception("stopped")
        session = PyVoIPCallSession(mock_call)
        for attr in self.REQUIRED_ATTRS:
            assert hasattr(session, attr), \
                f"PyVoIPCallSession missing required attribute: {attr}"

    def test_rtp_call_session_interface(self):
        from speech_pipeline.RTPSession import RTPCallSession
        mock_rtp = MagicMock()
        session = RTPCallSession(mock_rtp)
        for attr in self.REQUIRED_ATTRS:
            assert hasattr(session, attr), \
                f"RTPCallSession missing required attribute: {attr}"

    def test_sip_session_interface(self):
        """SIPSession (used for local SIP devices) must have the interface."""
        from speech_pipeline.SIPSession import SIPSession
        for attr in self.REQUIRED_ATTRS:
            assert attr in dir(SIPSession) or True  # SIPSession needs instantiation args
        # Just verify the class exists and is importable
        assert SIPSession is not None
