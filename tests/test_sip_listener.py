"""Tests for sip_listener._wait_for_bridge — attribute names and behavior."""
import inspect
import threading
from unittest.mock import MagicMock

from speech_pipeline.telephony import sip_listener, leg as leg_mod
from speech_pipeline.telephony.leg import Leg


class TestWaitForBridgeAttributes:

    def test_uses_sip_session_without_underscore(self):
        """_wait_for_bridge must access leg.sip_session, not leg._sip_session."""
        source = inspect.getsource(sip_listener._wait_for_bridge)
        assert "'_sip_session'" not in source and '"_sip_session"' not in source

    def test_leg_init_sets_sip_session(self):
        leg = Leg("l1", "inbound", "+49123", "pbx1", "sub1")
        assert hasattr(leg, "sip_session")
        assert leg.sip_session is None


class TestWaitForBridgeBehavior:

    def test_detects_hungup_session(self):
        """When sip_session.hungup is set, _wait_for_bridge should
        mark the leg completed and return."""
        leg = Leg("l-wfb-test", "inbound", "+49123", "pbx1", "sub1")
        leg_mod._legs[leg.leg_id] = leg

        session = MagicMock()
        session.hungup = threading.Event()
        session.hungup.set()
        leg.sip_session = session

        sip_listener._wait_for_bridge(leg, None)
        assert leg.status == "completed"

        leg_mod._legs.pop(leg.leg_id, None)

    def test_returns_immediately_when_bridged(self):
        """When call_id is set (leg was bridged), return without waiting."""
        leg = Leg("l-wfb-bridged", "inbound", "+49123", "pbx1", "sub1")
        leg.call_id = "some-call-id"
        leg_mod._legs[leg.leg_id] = leg

        # Should return immediately (call_id is set)
        sip_listener._wait_for_bridge(leg, None)
        # Status should NOT be changed to completed since it was bridged
        assert leg.status != "completed"

        leg_mod._legs.pop(leg.leg_id, None)


class TestSessionInterfaceConsistency:
    """All session types used by SIPSource/SIPSink must expose
    connected, hungup, rx_queue."""

    REQUIRED_ATTRS = ["connected", "hungup", "rx_queue"]

    def test_pyvoip_call_session(self):
        from speech_pipeline.telephony.leg import PyVoIPCallSession
        mock_call = MagicMock()
        mock_call.read_audio.side_effect = Exception("stopped")
        session = PyVoIPCallSession(mock_call)
        for attr in self.REQUIRED_ATTRS:
            assert hasattr(session, attr), f"PyVoIPCallSession missing {attr}"

    def test_rtp_call_session(self):
        from speech_pipeline.RTPSession import RTPCallSession
        session = RTPCallSession(MagicMock())
        for attr in self.REQUIRED_ATTRS:
            assert hasattr(session, attr), f"RTPCallSession missing {attr}"
